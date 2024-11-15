import asyncio
import tornado
import pickle
import base64
from utils import *
from tornado.queues import Queue
from tornado.concurrent import Future
import numpy as np
import datetime
import torch
import cv2
from collections import defaultdict
import logging
import pprint
import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

""" slices = {'vit_s0':[],'vit_s1':[], 'vit_s2':[],'vit_s3':[],'vit_large_s0':[],'vit_large_s1':[], 'vit_large_s2':[], 'vit_large_s3':[], 
          'swin_large_s0':[], 'swin_large_s1':[],'swin_large_s2':[],'swin_large_s3':[],'swin_s0':[],'swin_s1':[],'swin_s2':[],'swin_s3':[], 
          'deit_s0':[],'deit_s1':[],'deit_s2':[],'deit_s3':[]} """
total_slice = 4
req_queue = defaultdict(Queue)  # Separate queue for each slice
batch_size = 2
slices =defaultdict(list)
# Function to get a specific slice of a model

def load_slices(model_name):
    global slices
    for slc in range(total_slice):
        serving_tag = f'{model_name}_s{slc}'
        # Create and store slices based on the model type
        if 'swin' in serving_tag:
            if "swin_large" in serving_tag:
                slices[serving_tag] = Split_2_Swin(swin_large, slc).get_parts()
            else:
                slices[serving_tag] = Split_2_Swin(swin, slc).get_parts()
        else:
            if "vit_large" in serving_tag:
                slices[serving_tag] = Split_2_Vit(vit_large, slc).get_parts()
            elif "vit" in serving_tag:
                slices[serving_tag] = Split_2_Vit(vit, slc).get_parts()
            else:
                slices[serving_tag] = Split_2_Vit(deit, slc).get_parts()


# Prepare models and start serving
async def prepare_models_and_start_serving(model_name):
    logging.info(f"Entered prepare_models_and_start_serving... for {model_name}")
    try:
        global slices, req_queue

        logging.info(f"Preparing model {model_name}")

        load_slices(model_name)
          
        for slice_index in range(total_slice):
            serving_tag = f'{model_name}_s{slice_index}'
            req_queue[serving_tag] = Queue()
            logging.info(f"Initializing queue for {serving_tag}")

            tornado.ioloop.IOLoop.current().add_callback(process_request, model_name, slice_index)
            logging.info(f"Registered callback for {serving_tag}")
    except Exception as e:
        logging.error(f"Error in prepare_models_and_start_serving: {e}")
        sys.exit(1)

async def process_request(model_name: str, slice_index: int):
    global req_queue

    batch = []
    serving_tag = model_name + '_s' + str(slice_index)
    logging.info(f"Started processing for {serving_tag}...")
    try:
        while True:
            logging.info(f"Waiting for requests in {serving_tag}...")
            req = await req_queue[serving_tag].get()
            logging.info(f"Processing request from {serving_tag}")

            _, _, model_name, slice_req, _, _ = req

            batch.append(req)
            if len(batch) == batch_size:
                logging.info(f'Batch execution starts for model {model_name} slice {slice_req}')
                process_batch(batch, model_name, slice_index)
                logging.info(f'Batch execution ends for model {model_name} slice {slice_req}')
                req_queue[serving_tag].task_done()
                batch.clear()
    except Exception as e:
        logging.error(f"Error in process_request: {str(e)}")
        sys.exit(1)


# Process a batch of requests
def process_batch(batch, model_name, slice_index):
    global slices
    slc = slice_index
    serving_tag = f'{model_name}_s{slc}'
    logging.info(f'Processing {len(batch)} inputs for model {model_name} at slice {slc}')

    model_slice = slices[serving_tag][-1]

    # Prepare batch of images
    imgs = [req[4].unsqueeze(0) if req[4].dim() == 2 else req[4] for req in batch]
    imgs = torch.stack(imgs)
    if slc != 0:
        imgs = imgs.squeeze(1)

    # Process the batch through the model slice
    start_time = datetime.datetime.now()
    results = model_slice(imgs)
    end_time = datetime.datetime.now()
    time_batch = (end_time - start_time).total_seconds()

    for i, result in enumerate(results):
        client_id, req_id, model_name, slice_req, _, reply_future = batch[i]
        logging.info(f'Processed input {i} for client {client_id}, request {req_id}, slice {slc}')
        
        reply_data = {
            'client_id': client_id,
            'result': result,  # Assuming result is a tensor
            'model_name': model_name,
            'time_batch': time_batch,
            'batch': batch_size,
            'req_id': req_id,
            'slice': slc
        }
        reply_future.set_result(reply_data)
        logging.info(f"Returing result to client {client_id}, request {req_id}\n\n")

# Tornado request handler
class MainHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            # Decode and deserialize the request
            received_data = base64.b64decode(self.request.body)
            req = pickle.loads(received_data)

            client_id = req['client_id']
            req_id = req['request_id']
            model_name = req['model_name']
            slice_req = int(req['slice'])
            img = req['image']

            logging.info(f"Received request {req_id} from client: {client_id} for model: {model_name}, slice {slice_req}")

            reply_future = Future()
            serving_tag = model_name + "_s" + str(slice_req)

            # Put the request into the corresponding queue
            logging.info(f"Adding request to queue {serving_tag}")
            req_queue[serving_tag].put((client_id, req_id, model_name, slice_req, img, reply_future))
            logging.info(f"Request added to {serving_tag}")

            # Await the response for the final slice
            result = await reply_future
            logging.info(f"Received result for request {req_id} from {serving_tag}")

            response = {
                'client_id': result.get('client_id'),
                'result': result.get('result').tolist(),
                'model_name': result.get('model_name'),
                'batch': result.get('batch'),
                'time_batch': result.get('time_batch'),
                'req_id': result.get('req_id'),
                'slice': result.get('slice')
            }

            # Send the response to the client
            self.write(pickle.dumps(response))
            logging.info(f"Sent result {response['req_id']} from client: {response['client_id']} for model: {response['model_name']}, slice {response['slice']}")
        except Exception as e:
            logging.error(f"Error in post handler: {str(e)}")
            sys.exit(1)

# Create and start the Tornado application
def make_app():
    return tornado.web.Application([(r"/", MainHandler)])

async def main():
    logging.info("Starting main...")
    #prepare_slices(models_names, slices_lst)
    
    # Ensure callback is registered
    model_name = sys.argv[1]
    logging.info(f"Registering prepare_models_and_start_serving callback for {model_name}...")
    tornado.ioloop.IOLoop.current().add_callback(prepare_models_and_start_serving, model_name)

    # Start the Tornado app
    app = make_app()
    app.listen(8080)
    logging.info("Server is ready and listening on port 8080...")
    
    while True:
        await asyncio.sleep(3600)  # Keep the loop alive


if __name__ == "__main__":
    asyncio.run(main())
