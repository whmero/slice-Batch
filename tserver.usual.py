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
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

slices = {}
# Create a dictionary to hold requests based on client_id
client_requests = defaultdict(list)

# Queues for each slice
req_queue = Queue()
batch_size = 4

def get_slice(model_name, slc):
    global slices
    lst = []
    new_name = model_name+str(slc)
    if new_name in slices.keys():
            return slices[new_name][slc - 1]
    if 'swin' in model_name:
        if model_name == "swin_large":
            lst = Split_2_Swin(swin_large, slc).get_parts()
        else:
            lst = Split_2_Swin(swin, slc).get_parts()
    else:
        if model_name == "vit_large":
            lst = Split_2_Vit(vit_large, slc).get_parts()
        elif model_name == "vit":
            lst = Split_2_Vit(vit, slc).get_parts()
        else:
            lst = Split_2_Vit(deit, slc).get_parts()
    slices[new_name] = lst
    return lst[1]

async def process_request():
    try:
        async for req in req_queue:
            _, _, model_name, slice_req, _, _ = req
            tag = model_name + "_s" + str(slice_req)
            client_requests[tag].append(req)
            if len(client_requests[tag]) == batch_size:
                logging.info(f'Batch execution starts for slice {slice_req}')
                process_batch(client_requests[tag], slice_req)
                logging.info(f'Batch execution ends for slice {slice_req}')
                client_requests[tag].clear()
    except Exception as e:
        logging.error(f"Error in process_request: {str(e)}")


def process_batch(batch, slice_idx):
    logging.info(f'Processing {len(batch)} inputs at slice {slice_idx}')
    model_name = batch[0][2]
    model_slice = get_slice(model_name, slice_idx)

    # Load the images for this batch
    imgs = [req[4].unsqueeze(0) if req[4].dim() == 2 else req[4] for req in batch]
    
    imgs = torch.stack(imgs)
    if slice_idx != 1: 
        imgs = imgs.squeeze(1) 
    print(imgs.shape)
    # Process the batch through the model slice
    start_time = datetime.datetime.now()
    results = model_slice(imgs)
    end_time = datetime.datetime.now()
    time_batch = (end_time - start_time).total_seconds()

    for i, result in enumerate(results):
        client_id, req_id, model_name, slice_req, _, reply_future = batch[i]

        # Prepare the response data
        reply_data = {
            'client_id': client_id,
            'result': result,  # Assuming result is a tensor
            'model_name': model_name,
            'time_batch': time_batch,
            'batch': batch_size,
            'req_id': req_id,
            'slice': slice_idx
        }

        reply_future.set_result(reply_data)
        req_queue.task_done()

        logging.info(f'Done input {i} for client {client_id}, request {req_id}, slice {slice_idx}')


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
            logging.info(f"Received request {req_id} from client: {client_id} for model: {model_name}, slice {slice_req}")
            
            img = req['image']
            
            reply_future = Future()
            req_queue.put((client_id, req_id, model_name, slice_req, img, reply_future))

            # Process each slice up to the 4th one
            result = await reply_future
            reply_future = Future()            
            # Final slice (slice 4), prepare the response to send
            response = {
                'client_id': result.get('client_id'),
                'result': result.get('result').tolist(),  # Collect all results
                'model_name': result.get('model_name'),
                'batch': result.get('batch'),
                'time_batch': result.get('time_batch'),
                'req_id': result.get('req_id'),
                'slice': result.get('slice')
            }
            
            # Send the final accumulated response
            self.write(pickle.dumps(response))
            logging.info(f"Sent result {response['req_id']} from client: {response['client_id']} for model: {response['model_name']}, slice {response['slice']}")


        except Exception as e:
            logging.error(f"Error in post handler: {str(e)}")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

async def main():
    tornado.ioloop.IOLoop.current().add_callback(process_request)
    app = make_app()
    app.listen(8080)
    logging.info("server is ready...........................................................")
    # Block indefinitely without waiting on events that could trigger the shutdown
    while True:
        await asyncio.sleep(3600)  # Sleep for 1 hour in a loop


if __name__ == "__main__":
    asyncio.run(main())
