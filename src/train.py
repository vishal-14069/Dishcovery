
import argparse

import torch
import torchvision
from torchvision import transforms
from torch import nn

import data_setup, engine,model,utils
from src.helper_functions import set_device, set_seeds

# Create a parser 
parser = argparse.ArgumentParser(description="Get some hyperparameters")

# Get an arg for num_epochs 
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="the number of epochs to train for")

# Get an arg for batch_size
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")
# Get an arg for learning rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning to use during training")
# Get an arg for training dir
parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train/",
                    type=str,
                    help="specify the training directory")
# Get an arg for testing dir
parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test/",
                    type=str,
                    help="Specify the training directory")
#Device
parser.add_argument("--device",
                    default= "cpu",
                    type=str,
                    help="Speicfy the device")

args= parser.parse_args()


NUM_EPOCHS=args.num_epochs
BATCH_SIZE= args.batch_size
HIDDEN_UNITS= args.hidden_units
LEARNING_RATE= args.learning_rate
DEVICE= args.device

print(f"The Model will train for {NUM_EPOCHS} epochs with a batch size {BATCH_SIZE} with a learning rate of {LEARNING_RATE}")


# Setup Directories for Train and Test
train_dir= args.train_dir
test_dir= args.test_dir
print(f"train directory: {train_dir} | test directory: {test_dir}")

#device agnostic code

device = set_device(DEVICE)


# Setup the data transform
data_transform= transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])

embedding_dimension= 768 # vit-b embedding dimension
# Create the dataloaders
train_data,test_data,class_names= data_setup.create_dataloaders(train_dir=train_dir,test_dir=test_dir,
                                                                transform=data_transform,batch_size=BATCH_SIZE)


pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

pretrained_vit= torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device) 


for params in pretrained_vit.parameters():
    params.requires_grad=False

set_seeds()


pretrained_vit.heads= torch.nn.Linear(in_features=embedding_dimension,out_features=len(class_names),device=device)

# Setup Loss and Optimizer 

loss= nn.CrossEntropyLoss()
optimizer_fn= torch.optim.Adam(pretrained_vit.parameters(),lr=LEARNING_RATE,weight_decay=0.03)

# Setup the data engine to train and inference 

engine.train(model=pretrained_vit,
             train_dataloader=train_data,
             test_dataloader=test_data,
             optimizer=optimizer_fn,
             loss_fn=loss,
             epochs=NUM_EPOCHS,
             device=device)


utils.save_model(pretrained_vit,target_dir="models",model_name="trained_ViT.pth")
