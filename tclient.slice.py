import tornado.httpclient as httpclient
from tornado import ioloop, gen
import os
import sys
import cv2
import base64
import numpy as np
import datetime
import logging
from utils import *
import pickle
import torch
import requests

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
# Directory containing images
client_c = sys.argv[1]
model_name = str(sys.argv[2])
slice = int(sys.argv[3])
sending_start_time = None
receiving_end_time = None
total_inference_time = None
batch_type = 'slice'
if model_name == 'swin':
    model= Split_2_Swin(swin, slice)
    # Get the 4 separate parts of the model
    s1, s2 = model.get_parts()
elif model_name == 'swin_large':
    # Instantiate the split model class
    model= Split_2_Swin(swin_large, slice)
    # Get the 4 separate parts of the model
    s1, s2 = model.get_parts()
elif model_name == 'vit_large':
    model = Split_2_Vit(vit_large, slice)
    # Get the 4 separate parts of the model
    s1, s2 = model.get_parts()
elif model_name == 'deit':
    model = Split_2_Vit(deit, slice)
    # Get the 4 separate parts of the model
    s1, s2 = model.get_parts()
elif model_name == 'vit':
    model = Split_2_Vit(vit, slice)
    # Get the 4 separate parts of the model
    s1, s2 = model.get_parts()

async def send_request(req_id, http_client, img, client_c, model_name, slice):
    try:
        client_start_time = datetime.datetime.now()
        logging.info(f"Data preprocessing started")
        # Preprocess the image (resize, permute, and add batch dimension if needed)
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to (3, 224, 224)
        if slice != 0:
            img = img.unsqueeze(0)  # Add batch dimension: (1, 3, 224, 224)
            startP_time = datetime.datetime.now()
            with torch.no_grad():
                img = s1(img)
            endP_time = datetime.datetime.now()
            timeP = (endP_time - startP_time).total_seconds()
            logging.info(f"Data was processed by client model in time {timeP}")
        else:
            timeP = 0
            logging.info(f"Data will be sent without processing")

        # Create the request data
        post_data = {
            'client_id': client_c,
            'request_id': req_id + 1,
            'image': img,
            'model_name': model_name,
            'slice': slice
        }

        # Serialize and encode the data
        sending_start_time = datetime.datetime.now()
        serialized_data = pickle.dumps(post_data)
        body = base64.b64encode(serialized_data)
        sending_start_time = datetime.datetime.now()
        logging.info(f"Sent request {post_data['request_id']} with slice {post_data['slice']} result for client {post_data['client_id']}")
        response = await http_client.fetch("http://192.168.84.116:8080", method  ='POST', headers = None, body = body, request_timeout = 300)
        # Measure times
        receiving_end_time = datetime.datetime.now()
        client_exec_time = (sending_start_time - client_start_time).total_seconds()
        server_exec_time = (receiving_end_time - sending_start_time).total_seconds()
        total_inference_time = (receiving_end_time - client_start_time).total_seconds()
        # Send the request asynchronously
        response_data = pickle.loads(response.body)
        logging.info(f"Received slice {response_data['slice']} result for client {response_data['client_id']}")
        if len(response.body)!=0:
            # Handle the response
            handle_response(response_data, slice, client_exec_time, server_exec_time, total_inference_time, timeP, client_c, batch_type)

    except httpclient.HTTPError as e:
        logging.error(f"HTTPError occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        
async def main():
    http_client = httpclient.AsyncHTTPClient()

    # Sending multiple requests concurrently
    for req_id in range(int(len(imgs_path) / 2)):

        img = cv2.imread(imgs_dir + imgs_path[req_id])
        logging.info(f"Data was read for processing")

        # Call the request function asynchronously
        ioloop.IOLoop.current().add_callback(send_request, req_id, http_client, img, client_c, model_name, slice)

        # Optional delay to avoid overwhelming the server, adjust this as needed
        await gen.sleep(1)

    # Close the HTTP client when done
    http_client.close()

if __name__ == '__main__':
    io_loop = ioloop.IOLoop.current()
    io_loop.add_callback(main)
    io_loop.start()
