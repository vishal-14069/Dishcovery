
"""Trains a PyTorch Model using device agnostic code"""

import os
import torch
from torchvision import transforms 
from timeit import default_timer as timer 
import data_setup,engine,model,utils

# Setup the hyper-parameters 
NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

# Setup the train and test directories 

train_dir= "data/pizza_steak_sushi/train"
test_dir= "data/pizza_steak_sushi/test"

#Setup Device Agnostic Code

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Setup the data transform 

data_transform= transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# Setup the train data loader and the test dataloader using the data_setup script

train_dataloader,test_dataloader,class_names=data_setup.create_dataloaders(train_dir=train_dir,
                                                                            test_dir=test_dir,
                                                                            transform=data_transform,
                                                                            batch_size=BATCH_SIZE)
# Create model

model_1 = model.TinyVGG(input_shape=3,hidden_units=HIDDEN_UNITS,output_shape=len(class_names)).to(device)

# Setup Loss and Optimizer 

loss_fn= torch.nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model_1.parameters(),lr=LEARNING_RATE)

start_time = timer()

engine.train(model=model_1,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             device=device,
             epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model

utils.save_model(model=model_1,target_dir="models",model_name= "going_modular_script_model.pth")
