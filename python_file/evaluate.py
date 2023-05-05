
import os
import torch

from timeit import default_timer as timer 
from python_file import data_setup, traintest, tinyvggmodel, saving

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.0001

# Setup directories
train_dir = "data/5_Classes_of_food_101/train"
test_dir = "data/5_Classes_of_food_101/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=32,
    num_workers=os.cpu_count()
)

# Create model with help from tinyvggmodel.py
model = tinyvggmodel.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

start_time = timer()

# Start training with help from traintest.py
traintest.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")     

#save the model

saving.save_model(model=model,
               target_dir="models",
               model_name="food101_model_with_tinyvgg.pth")

