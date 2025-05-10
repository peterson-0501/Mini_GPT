"""
Training script for the Bigram or MiniGPT Language Model with training and validation sets.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb


from model_1 import BigramLanguageModel, MiniGPT  # Import both models
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig  # Import both configs

# Define which model to use: "bigram" or "minigpt"
MODEL = "minigpt"

# Load the configuration and model based on the selected MODEL
if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")

# Adjust configuration parameters as needed
config.batch_size = 32  # Batch size
config.to_clip_grad = True  # Clip gradients
config.gradient_clip = 0.5  # Gradient clipping threshold
config.max_epochs = 10  # Number of training epochs
config.log_interval = 100  # Interval to log training loss
config.save_iterations = 1  # Save model every `save_iterations` epochs
config.max_iterations = 1500  # Maximum iterations for training


config.num_heads = 8

config.save_path = Path("models/minigpt_scaled/")  

  # Maximum iterations for evaluation per epoch


# Initialize WandB if logging is enabled
if config.to_log:
    wandb.init(project="dl2_proj3_minigpt")
    
#change it from minigt=pt to config
if not Path.exists(config.save_path):
    Path.mkdir(config.save_path, parents=True, exist_ok=True)

# Function to count trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load the dataset and split into training and validation sets
train_data = TinyStoriesDataset(config.path_to_data, mode="train", context_length=config.context_length)
val_data = TinyStoriesDataset(config.path_to_data, mode="test", context_length=config.context_length)

# DataLoader initialization
train_dataloader = DataLoader(train_data, batch_size=config.batch_size, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size=config.batch_size, pin_memory=True)

# Device selection
device = torch.device("cpu")
model.to(device)

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop

for epoch in range(1, config.max_epochs + 1):
    model.train()
    total_loss = 0.0
    
    iteration_count = 0
    for batch_idx, batch_data in enumerate(train_dataloader):
        
        # Check if maximum iterations reached
        if iteration_count >= config.max_iterations:
            break
        
        inputs, targets = batch_data
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)  #logits, input goes to the forward fucntion of bigram model
        loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1)) ##criterion: This is an instance of nn.CrossEntropyLoss()  #model.vocab_size) change it to config.vocab_size
        
        loss.backward()
         
        # Gradient clipping #his helps to stabilize training and prevent large updates that can disrupt learning.Gradient clipping is a technique used to prevent exploding gradients in deep neural networks during training
        if config.to_clip_grad: 
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        
        optimizer.step() ##Once gradients are computed, optimizer.step() updates the parameters of the model using the gradients.

        
        total_loss += loss.item()
        iteration_count += 1
        
        # Logging training loss
        if iteration_count % config.log_interval == 0:
            avg_loss = total_loss / config.log_interval
            print(f"Iteration [{iteration_count}/{config.max_iterations}], Epoch [{epoch}/{config.max_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Training Loss: {avg_loss:.4f}")
            
            if config.to_log:
                wandb.log({"Training Loss": avg_loss})
            
            total_loss = 0.0
        
    
    
    # Validation after each epoch
    model.eval()
    eval_loss = 0.0
    eval_iterations = 0
    with torch.no_grad():
        for eval_batch_idx, eval_batch_data in enumerate(val_dataloader):
            eval_inputs, eval_targets = eval_batch_data
            eval_inputs, eval_targets = eval_inputs.to(device), eval_targets.to(device)
            
            eval_outputs = model(eval_inputs)
            loss = criterion(eval_outputs.view(-1, config.vocab_size), eval_targets.view(-1))  #model.vocab_size) change it to config.vocab_size
            eval_loss += loss.item()
            eval_iterations += 1
            
            if eval_iterations >= 100:
                break
        
        eval_loss /= eval_iterations
        
        print(f"Epoch [{epoch}/{config.max_epochs}], Validation Loss: {eval_loss:.4f}")
        
        # Logging validation loss
        if config.to_log:
            wandb.log({"Validation Loss": eval_loss})
    
    # Save model checkpoint
    if epoch % config.save_iterations == 0:
        torch.save(model.state_dict(), config.save_path / f"{MODEL}_epoch_{epoch}.pt")

print("Training completed.")
