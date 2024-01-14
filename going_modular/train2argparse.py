
#### Let's use argparse to allow us to specify the hyper-parameters on the command line 

import os
import argparse

import torch

from torchvision import transforms
from torch import nn

import data_setup, engine,model,utils

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
# Get an arg for hidden_units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="Number of Hidden Units")
# Get an arg for learning rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning to use during training")
# Get an arg for training dir
parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train",
                    type=str,
                    help="specify the training directory")
# Get an arg for testing dir
parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test",
                    type=str,
                    help="Specify the training directory")


# get our arguments for test directory
args= parser.parse_args()

# Setup hyper-parameters
NUM_EPOCHS=args.num_epochs
BATCH_SIZE= args.batch_size
HIDDEN_UNITS= args.hidden_units
LEARNING_RATE= args.learning_rate

print(f"The Model will train for {NUM_EPOCHS} epochs with a batch size {BATCH_SIZE} using {HIDDEN_UNITS} layers with a learning rate of {LEARNING_RATE}")


# Setup Directories for Train and Test
train_dir= args.train_dir
test_dir= args.test_dir
print(f"train directory: {train_dir} | test directory: {test_dir}")

#device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"


# Setup the data transform
data_transform= transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

# Create the dataloaders
train_data,test_data,class_names= data_setup.create_dataloaders(train_dir=train_dir,test_dir=test_dir,
                                                                transform=data_transform,batch_size=32)

# Instantiate the model
model2= model.TinyVGG(input_shape=3,hidden_units=HIDDEN_UNITS,output_shape=len(class_names)).to(device)
# Setup Loss and Optimizer 

loss= nn.CrossEntropyLoss()
optimizer_fn= torch.optim.Adam(model2.parameters(),lr=LEARNING_RATE)

# Setup the data engine to train and inference 

engine.train(model=model2,
             train_dataloader=train_data,
             test_dataloader=test_data,
             optimizer=optimizer_fn,
             loss_fn=loss,
             epochs=NUM_EPOCHS,device=device)

# Let's save the model

utils.save_model(model2,target_dir="models",model_name="ArgParse_TinyVGG.pth")
