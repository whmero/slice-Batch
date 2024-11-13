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
import logging
from collections import defaultdict

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Initialize global variables
client_requests = defaultdict(list)
slices = {'vit_s0':[],'vit_s1':[], 'vit_s2':[],'vit_s3':[],'vit_large_s0':[],'vit_large_s1':[], 'vit_large_s2':[], 'vit_large_s3':[], 
          'swin_large_s0':[], 'swin_large_s1':[],'swin_large_s2':[],'swin_large_s3':[],'swin_s0':[],'swin_s1':[],'swin_s2':[],'swin_s3':[], 
          'deit_s0':[],'deit_s1':[],'deit_s2':[],'deit_s3':[]}
total_slice = 3
req_queue = defaultdict(Queue)  # Separate queue for each slice
batch_size = 4

# Function to get a specific slice of a model
def get_slices(model_name):
    global slices
    # Create and store slices based on the model type
    if 'swin' in model_name:
        if "swin_large" in model_name:
            slices[model_name] = SplitSwin(swin_large, split_idx1, split_idx2, split_idx3).get_parts()
        else:
            slices[model_name] = SplitSwin(swin, split_idx1, split_idx2, split_idx3).get_parts()
    else:
        if "vit_large" in model_name:
            slices[model_name] = SplitViT(vit_large, split_idx1, split_idx2, split_idx3).get_parts()
        elif "vit" in model_name:
            slices[model_name] = SplitViT(vit, split_idx1, split_idx2, split_idx3).get_parts()
        else:
            slices[model_name] = SplitViT(deit, split_idx1, split_idx2, split_idx3).get_parts()

# Prepare models and start serving
async def prepare_models_and_start_serving():
    logging.info("Entered prepare_models_and_start_serving...")
    try:
        global slices, req_queue

        logging.info(f"Preparing models {slices.keys()}")

        for model_name in slices.keys():
            get_slices(model_name)
            logging.info(f"Initialized slices for {model_name}")

        for model_name in slices.keys():
            for slice_index in range(total_slice):
                serving_tag = model_name
                req_queue[serving_tag] = Queue()
                logging.info(f"Initializing queue for {serving_tag}")

                tornado.ioloop.IOLoop.current().add_callback(process_request, model_name, slice_index)
                logging.info(f"Registered callback for {serving_tag}")
    except Exception as e:
        logging.error(f"Error in prepare_models_and_start_serving: {e}")

async def process_request(model_name: str, slice_index: int):
    serving_tag = model_name
    logging.info(f"Started processing for {serving_tag}...")
    try:
        while True:
            logging.info(f"Waiting for requests in {serving_tag}...")
            req = await req_queue[serving_tag].get()
            logging.info(f"Processing request from {serving_tag}: {req}")

            _, _, model_name, slice_req, _, _ = req

            client_requests[serving_tag].append(req)
            if len(client_requests[serving_tag]) == batch_size:
                logging.info(f'Batch execution starts for slice {slice_req}')
                process_batch(client_requests[serving_tag], serving_tag)
                logging.info(f'Batch execution ends for slice {slice_req}')
                client_requests[serving_tag].clear()
    except Exception as e:
        client_requests[serving_tag].clear()
        logging.error(f"Error in process_request: {str(e)}")


# Process a batch of requests
def process_batch(batch, serving_tag):
    global slices
    slc = int(serving_tag[-1])
    logging.info(f'Processing {len(batch)} inputs at slice {slc}')
    model_name = batch[0][2]

    model_slice = slices[serving_tag][slc]

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

        if slc < total_slice - 1:  # Not the last slice
            next_slice_idx = slc + 1
            next_serving_tag = model_name + "_s" + str(next_slice_idx)
            logging.info(f"Forwarding result to {next_serving_tag}")
            req_queue[next_serving_tag].put((client_id, req_id, model_name, next_slice_idx, result, reply_future))
        else:  # Final slice
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
            req_queue[serving_tag].task_done()

        logging.info(f'Processed input {i} for client {client_id}, request {req_id}, slice {slc}')

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
            await req_queue[serving_tag].put((client_id, req_id, model_name, slice_req, img, reply_future))
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

# Create and start the Tornado application
def make_app():
    return tornado.web.Application([(r"/", MainHandler)])

async def main():
    logging.info("Starting main...")
    #prepare_slices(models_names, slices_lst)
    
    # Ensure callback is registered
    logging.info("Registering prepare_models_and_start_serving callback...")
    tornado.ioloop.IOLoop.current().add_callback(prepare_models_and_start_serving)

    # Start the Tornado app
    app = make_app()
    app.listen(8080)
    logging.info("Server is ready and listening on port 8080...")
    
    while True:
        await asyncio.sleep(3600)  # Keep the loop alive


if __name__ == "__main__":
    asyncio.run(main())
