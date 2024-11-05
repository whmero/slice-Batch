import torch
import torch.nn as nn
import timm
import pickle
import csv
import logging
import os

imgs_dir = "Images/"
imgs_path = os.listdir(imgs_dir)
metrics_headers = ['client_id','req_id', 'model_name', 'batch_type', 'batch', 'slice', 'time_batch','client_exec_time', 'server_exec_time', 'total_inference_time']
# Load the pre-trained Swin Transformer model
swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
# Load the pre-trained Swin Transformer model using timm's create_model function
swin_large = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
# Load the pre-trained ViT large model using timm's create_model function
vit_large = timm.create_model('vit_large_patch16_224', pretrained=True)
# Load the pre-trained DeiT model using timm's create_model function
deit = timm.create_model('deit_base_patch16_224', pretrained=True)
# Load the pre-trained ViT model using timm's create_model function
vit = timm.create_model('vit_base_patch16_224', pretrained=True)

# Define split indices for transformer blocks
split_idx1 = 3   # After the first 3 transformer blocks
split_idx2 = 6   # After the next 3 transformer blocks
split_idx3 = 9   # After the next 3 transformer blocks 

# Custom function to split Swin Transformer into 4 separate models
class SplitSwin:
    def __init__(self, model, split_idx1, split_idx2, split_idx3):
        # Part 1: Initial layers and first set of transformer blocks
        self.part1 = nn.Sequential(
            model.patch_embed,      # Patch embedding
            *model.layers[:split_idx1]  # First group of Swin blocks
        )
        # Part 2: Middle transformer blocks
        self.part2 = nn.Sequential(
            *model.layers[split_idx1:split_idx2]  # Second group of Swin blocks
        )
        # Part 3: Later transformer blocks
        self.part3 = nn.Sequential(
            *model.layers[split_idx2:split_idx3]  # Third group of Swin blocks
        )
        # Part 4: Final transformer blocks and classification head
        self.part4 = nn.Sequential(
            *model.layers[split_idx3:],  # Final group of Swin blocks
            model.norm,                  # Layer norm
            model.head                   # Classification head
        )
    
    def get_parts(self):
        # Return all 4 parts as separate models
        return [self.part1, self.part2, self.part3, self.part4]
    
class SplitViT:
    def __init__(self, model, split_idx1, split_idx2, split_idx3):
        # Part 1: Embedding and initial transformer blocks
        self.part1 = nn.Sequential(
            model.patch_embed,  # Patch embedding
            model.pos_drop,     # Positional embedding dropout
            *model.blocks[:split_idx1]  # First group of transformer blocks
        )
        # Part 2: Middle transformer blocks
        self.part2 = nn.Sequential(
            *model.blocks[split_idx1:split_idx2]  # Second group of transformer blocks
        )
        # Part 3: Later transformer blocks
        self.part3 = nn.Sequential(
            *model.blocks[split_idx2:split_idx3]  # Third group of transformer blocks
        )
        # Part 4: Final transformer block and classification head
        self.part4 = nn.Sequential(
            *model.blocks[split_idx3:],  # Final transformer block
            model.norm,                  # Layer norm
            model.head                   # Classification head
        )
    
    def get_parts(self):
        # Return all 4 parts as separate models
        return [self.part1, self.part2, self.part3, self.part4]

class Split_2_Swin:
    def __init__(self, model, s):
        if s == 0:
            idx = 0
            # Part 1: Initial layers and first set of transformer blocks
            self.part1 = model
            # Part 2: Final transformer blocks and classification head
            self.part2 = None
        else:
            if s == 1:
                idx = split_idx1
            elif s == 2:
                idx = split_idx2
            else:
                idx = split_idx3
            # Part 1: Initial layers and first set of transformer blocks
            self.part1 = nn.Sequential(
                model.patch_embed,      # Patch embedding
                *model.layers[:idx]  # First group of Swin blocks
            )
            # Part 2: Final transformer blocks and classification head
            self.part2 = nn.Sequential(
                *model.layers[idx:],  # Final group of Swin blocks
                model.norm,                  # Layer norm
                model.head                   # Classification head
            )
    
    def get_parts(self):
        # Return all 4 parts as separate models
        return self.part1, self.part2

class Split_2_Vit:
    def __init__(self, model, s):
        if s == 0:
            idx = 0
            # Part 1: Initial layers and first set of transformer blocks
            self.part1 = model
            # Part 2: Final transformer blocks and classification head
            self.part2 = None
        else:
            if s == 1:
                idx = split_idx1
            elif s == 2:
                idx = split_idx2
            else:
                idx = split_idx3
            # Part 1: Initial layers and first set of transformer blocks
            self.part1 = nn.Sequential(
                model.patch_embed,  # Patch embedding
                model.pos_drop,     # Positional embedding dropout
                *model.blocks[:idx]  # First group of transformer blocks
            )
            # Part 2: Final transformer blocks and classification head
            self.part2 = nn.Sequential(
                *model.blocks[idx:],  # Final transformer block
                model.norm,                  # Layer norm
                model.head                   # Classification head
            )
    def get_parts(self):
        # Return all 4 parts as separate models
        return self.part1, self.part2

def write_to_csv(filename, field_names, data):
    # Check if the file exists
    file_exists = False
    try:
        with open(filename, 'r') as file:   
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in the appropriate mode
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        # Write a new line if the file is empty
        if not file_exists:
            writer.writerow(field_names)  # Example column headers

        # Write the data to the file
        writer.writerow(data)
        
def handle_response(data, client_exec_time, server_exec_time, total_inference_time, client_c, batch_type):
    try:
        
        logging.info("Received result of processing request no. "+ str(data['req_id']) + " in batching time = "+ str(data['time_batch'])+ " seconds from server with model: " + data['model_name']+ " and total processing time =  " + str(total_inference_time))
        write_to_csv("results/" + client_c + '.csv', metrics_headers, [data['client_id'], data['req_id'], data['model_name'], batch_type, data['batch'], str(data['slice']), data['time_batch'], str(client_exec_time), str(server_exec_time), str(total_inference_time)])

    except Exception as e:
        logging.error(f"Error decoding or unpickling the response: {str(e)}")