import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from transformers import ViTConfig, ViTForImageClassification
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.optim.lr_scheduler as lr_scheduler
import torch
from torch import nn
import time

class NPYImageDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames  

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = np.load(fname) 
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        label = 1 if fname.endswith('1.npy') else 0

        return img, torch.tensor(label, dtype=torch.long)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

def train_model(epochs=10, lr=1e-4, device='cuda'):

    config = ViTConfig(
        image_size=128,
        patch_size=16,
        num_channels=12,
        hidden_size=768,
        num_hidden_layers=3,
        num_attention_heads=3,
        intermediate_size=3072,
        num_labels=2,
    )

    model = ViTForImageClassification(config)

    training_files = os.listdir('SmallerSampleData/train_sample')
    validation_files = os.listdir('SmallerSampleData/validation_sample')

    base_path = 'SmallerSampleData/'
    training_files = [base_path + 'train_sample/' +  f for f in training_files]
    validation_files = [base_path + 'validation_sample/' + f for f in validation_files]

    print(f"Training files: {len(training_files)}")
    print(f"Validation files: {len(validation_files)}")

    device_count = torch.cuda.device_count()
    global_batch_size = 64 * device_count

    training_set = NPYImageDataset(
        filenames=training_files,
    )
    validation_set = NPYImageDataset(
        filenames=validation_files,
    )

    training_dataloader = DataLoader(
        training_set,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=global_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = model.to(device)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    data_dict = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    time_dict = {'epoch': [], 'epoch_start_time': [], 'epoch_end_time': [],
                  'training_set_start_time': [], 'training_set_end_time': [],
                  'training_iteration_start_time': [], 'training_iteration_end_time': [],
                  'training_to_device_start_time': [], 'training_to_device_end_time': [],
                  'training_forward_start_time': [], 'training_forward_end_time': [],
                  'training_backward_start_time': [], 'training_backward_end_time': [],
                  'validation_set_start_time': [], 'validation_set_end_time': [],
                  'validation_iteration_start_time': [], 'validation_iteration_end_time': [],
                  'validation_to_device_start_time': [], 'validation_to_device_end_time': [],
                  'validation_forward_start_time': [], 'validation_forward_end_time': [],
                }

    print(f"DataParallel is wrapping the model across GPUs: {device_count}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    for epoch in range(epochs):

        print(f"Epoch {epoch+1}/{epochs}")

        epoch_start_time = time.time()

        training_iteration_start_times = []
        training_iteration_end_times = []

        training_to_device_start_times = []
        training_to_device_end_times = []

        training_forward_start_times = []
        training_forward_end_times = []

        training_backward_start_times = []
        training_backward_end_times = []

        training_set_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        training_iteration_start_time = time.time()
        training_to_device_start_time = time.time()

        for images, labels in training_dataloader:

            images, labels = images.to(device), labels.to(device)

            training_to_device_end_time = time.time()

            training_forward_start_time = time.time()

            outputs = model(pixel_values=images)

            loss = loss_fn(outputs.logits, labels)

            training_forward_end_time = time.time()

            training_backward_start_time = time.time()

            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

            training_backward_end_time = time.time()

            train_loss += loss.item() * images.size(0)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            training_iteration_end_time = time.time()

            training_iteration_start_times.append(training_iteration_start_time)
            training_iteration_end_times.append(training_iteration_end_time)

            training_to_device_start_times.append(training_to_device_start_time)
            training_to_device_end_times.append(training_to_device_end_time)

            training_forward_start_times.append(training_forward_start_time)
            training_forward_end_times.append(training_forward_end_time)

            training_backward_start_times.append(training_backward_start_time)
            training_backward_end_times.append(training_backward_end_time)

            training_iteration_start_time = time.time()
            training_to_device_start_time = time.time()

       
        avg_train_loss = train_loss / total
        train_acc = correct / total

        training_set_end_time = time.time()

        validation_iteration_start_times = []
        validation_iteration_end_times = []

        validation_to_device_start_times = []
        validation_to_device_end_times = []

        validation_forward_start_times = []
        validation_forward_end_times = []

        validation_set_start_time = time.time()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        validation_iteration_start_time = time.time()
        validation_to_device_start_time = time.time()

        with torch.no_grad():
            for images, labels in validation_dataloader:
                
                images, labels = images.to(device), labels.to(device)

                validation_to_device_end_time = time.time()

                validation_forward_start_time = time.time()

                outputs = model(pixel_values=images)

                loss = loss_fn(outputs.logits, labels)

                validation_forward_end_time = time.time()

                val_loss += loss.item() * images.size(0)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                validation_iteration_end_time = time.time()

                validation_iteration_start_times.append(validation_iteration_start_time)
                validation_iteration_end_times.append(validation_iteration_end_time)

                validation_to_device_start_times.append(validation_to_device_start_time)
                validation_to_device_end_times.append(validation_to_device_end_time)

                validation_forward_start_times.append(validation_forward_start_time)
                validation_forward_end_times.append(validation_forward_end_time)

                validation_iteration_start_time = time.time()
                validation_to_device_start_time = time.time()

        validation_set_end_time = time.time()

        avg_val_loss = val_loss / total
        val_acc = correct / total

        epoch_end_time = time.time()

        data_dict['epoch'].append(epoch+1)
        data_dict['train_loss'].append(avg_train_loss)
        data_dict['train_acc'].append(train_acc)
        data_dict['val_loss'].append(avg_val_loss)
        data_dict['val_acc'].append(val_acc)

        time_dict['epoch'].append(epoch+1)
        time_dict['epoch_start_time'].append(epoch_start_time)
        time_dict['epoch_end_time'].append(epoch_end_time)
        time_dict['training_set_start_time'].append(training_set_start_time)
        time_dict['training_set_end_time'].append(training_set_end_time)
        time_dict['validation_set_start_time'].append(validation_set_start_time)
        time_dict['validation_set_end_time'].append(validation_set_end_time)
        time_dict['training_iteration_start_time'].append(training_iteration_start_times)
        time_dict['training_iteration_end_time'].append(training_iteration_end_times)
        time_dict['training_to_device_start_time'].append(training_to_device_start_times)
        time_dict['training_to_device_end_time'].append(training_to_device_end_times)
        time_dict['training_forward_start_time'].append(training_forward_start_times)
        time_dict['training_forward_end_time'].append(training_forward_end_times)
        time_dict['training_backward_start_time'].append(training_backward_start_times)
        time_dict['training_backward_end_time'].append(training_backward_end_times)
        time_dict['validation_iteration_start_time'].append(validation_iteration_start_times)
        time_dict['validation_iteration_end_time'].append(validation_iteration_end_times)
        time_dict['validation_to_device_start_time'].append(validation_to_device_start_times)
        time_dict['validation_to_device_end_time'].append(validation_to_device_end_times)
        time_dict['validation_forward_start_time'].append(validation_forward_start_times)
        time_dict['validation_forward_end_time'].append(validation_forward_end_times)


    data_df = pd.DataFrame(data_dict)
    time_df = pd.DataFrame(time_dict)
    device_count = torch.cuda.device_count()
    data_df.to_csv(f'training_data_pt_{device_count}gpus_opt.csv', index=False)
    time_df.to_csv(f'training_time_pt_{device_count}gpus_opt.csv', index=False)

def main():

    train_model(epochs=15, lr=1e-4, device='cuda')

    return

if __name__ == "__main__":
    main()