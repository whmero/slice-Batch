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

torch.backends.cudnn.benchmark = True
import requests

# Set device for PyTorch (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

client_c = sys.argv[1]
model_name = str(sys.argv[2])
slice = int(sys.argv[3])
sending_start_time = None
receiving_end_time = None
total_inference_time = None
batch_type = 'slice'

# Load the appropriate model and move it to GPU
if model_name == 'swin':
    model = Split_2_Swin(swin, slice)
elif model_name == 'swin_large':
    model = Split_2_Swin(swin_large, slice)
elif model_name == 'vit_large':
    model = Split_2_Vit(vit_large, slice)
elif model_name == 'deit':
    model = Split_2_Vit(deit, slice)
elif model_name == 'vit':
    model = Split_2_Vit(vit, slice)
else:
    raise ValueError(f"Unknown model name: {model_name}")

# Get model slices
s1, s2 = model.get_parts()
s1 = s1.to(device)

# Move model parts to GPU if available
s1 = s1.to(device)
s2 = s2.to(device)

async def send_request(req_id, http_client, img, client_c, model_name, slice):
    try:
        client_start_time = datetime.datetime.now()
        logging.info("Data preprocessing started")

        # Preprocess the image: resize, convert to tensor, move to device
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        if slice != 0:
            startP_time = datetime.datetime.now()
            with torch.no_grad():
                img = s1(img)  # Perform inference on GPU
            endP_time = datetime.datetime.now()
            timeP = (endP_time - startP_time).total_seconds()
            logging.info(f"Data processed by client model in {timeP} seconds")
        else:
            timeP = 0
            logging.info("Data will be sent without processing")

        # Prepare request data
        post_data = {
            'client_id': client_c,
            'request_id': req_id + 1,
            'image': img.cpu(),  # Move tensor to CPU before sending
            'model_name': model_name,
            'slice': slice
        }

        # Serialize and encode data
        sending_start_time = datetime.datetime.now()
        serialized_data = pickle.dumps(post_data)
        body = base64.b64encode(serialized_data)

        logging.info(f"Sent request {post_data['request_id']} with slice {post_data['slice']} for client {post_data['client_id']}")

        # Send request asynchronously
        response = await http_client.fetch("http://192.168.86.104:8080", method='POST', headers=None, body=body, request_timeout=300)

        # Measure times
        receiving_end_time = datetime.datetime.now()
        client_exec_time = (sending_start_time - client_start_time).total_seconds()
        server_exec_time = (receiving_end_time - sending_start_time).total_seconds()
        total_inference_time = (receiving_end_time - client_start_time).total_seconds()

        # Process response
        response_data = pickle.loads(response.body)
        logging.info(f"Received slice {response_data['slice']} result for client {response_data['client_id']}")

        if response.body:
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
        logging.info("Data was read for processing")

        # Send request asynchronously
        ioloop.IOLoop.current().add_callback(send_request, req_id, http_client, img, client_c, model_name, slice)

        # Optional delay to avoid overwhelming the server
        await gen.sleep(1)

    # Close HTTP client
    http_client.close()


if __name__ == '__main__':
    io_loop = ioloop.IOLoop.current()
    io_loop.add_callback(main)
    io_loop.start()
