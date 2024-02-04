
#### Let's use argparse to allow us to specify the hyper-parameters on the command line 

import sys

import argparse
import time
import torch
import torchvision

from helper_functions import set_device, set_seeds,pred_and_plot_image

# Create a parser 
parser = argparse.ArgumentParser(description="Get image path for prediction")

# Get an arg for num_epochs 

# Get an arg for batch_size
parser.add_argument("--image_path",
                    default=f"{sys.path[-1]}assets/",
                    type=str,
                    help="predict image path")
#Device
parser.add_argument("--device",
                    default= "cpu",
                    type=str,
                    help="Speicfy the device")

args= parser.parse_args()

DEVICE= args.device

embedding_dimension=768

weights= f"{sys.path[0]}/trained_ViT.pth"

device = set_device(DEVICE)
set_seeds()

images_dir= "/Users/vishal./data/food_classification_dataset"
class_names= torchvision.datasets.ImageFolder(images_dir).classes

model= torchvision.models.vit_b_16().to(device)

model.heads= torch.nn.Linear(in_features=embedding_dimension,out_features=len(class_names),device=device)

model.load_state_dict(torch.load("trained_ViT.pth"))

if __name__=='__main__':
    start_time= time.time()
    predict=pred_and_plot_image(model=model,image_path=args.image_path,class_names=class_names,mode='predict')
    end_time= time.time()
    print(predict)
    print(f"Time Taken for Inference: {end_time-start_time:.3f} seconds")



