
from EarlyStopper import *
from utils import *

import torch
import torch.optim as optim
from tqdm import trange
import os
import pandas
from pandas import DataFrame
import time
#train the model
def train(net, trainloader, valloader, EPOCHS, LEARNING_RATE, IMG_CHANNELS, IMG_SIZE, MODEL_NAME, patience, validate_every, device):
    
    optimizer = optim.Adam(net.parameters(), LEARNING_RATE, betas=(0.9, 0.999), eps=1e-09, weight_decay=0, amsgrad=False)
    weight = torch.tensor([2.0, 2.0, 1.5, 1.0])

    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    with open("model.log", "a") as f:
        for epoch in trange(EPOCHS):
                net.train()
                for batch_idx, samples in enumerate(trainloader):
                    x, y = samples

                    filter_input(x)

                    x = x.view(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
                    x = x.to(device)
                    y = y.to(device)

                    la = y[:, 0]
                    lb = y[:, 1]
                    ld = y[:, 2]
                    lt = y[:, 3]

                    a, b, d, t = net(x)
                    
                    loss = weighted_MSE(a, b, d, t, la, lb, ld, lt, weight, device)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                net.eval()
                for batch_idx, samples in enumerate(valloader):
                    x, y = samples

                    filter_input(x)

                    x = x.view(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
                    x = x.to(device)
                    y = y.to(device)

                    la = y[:, 0]
                    lb = y[:, 1]
                    ld = y[:, 2]
                    lt = y[:, 3]

                    a, b, d, t = net(x)
                    loss = weighted_MSE(a, b, d, t, la, lb, ld, lt, weight, device)
                    val_losses.append(loss.item())

                train_loss = np.average(train_losses)
                valid_loss = np.average(val_losses)
                avg_train_losses.append(train_loss)
                avg_val_losses.append(valid_loss)
                

                train_losses = []
                val_losses = []


                f.write(f"{MODEL_NAME},{round(time.time(), 3)},  {round(float(train_loss), 4)},  {round(float(valid_loss),4)}\n")
                print("\nloss : ", train_loss, "val loss : ", valid_loss, "\n")
                
                early_stopping(valid_loss, net)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        net.load_state_dict(torch.load('checkpoint.pt'))
        
    return avg_train_losses, avg_val_losses

def test(net, testloader, IMG_CHANNELS, IMG_SIZE, OUTPUT_LABEL_SIZE, device):
    predictions = []
    with torch.no_grad():
        for i, sample in enumerate(testloader):
            predict = []
            x ,y = sample

            filter_input(x)

            x = x.view(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
            
            x = x.to(device)
            y = y.to(device)
            
            a, b, d, t = net(x)


            final_result = torch.cat([a, b, d, t], dim = 1)
            final_result = final_result.to("cpu")
          
            predictions.append(final_result.numpy())

    print(predictions)
    
    predictions = np.array(predictions)
    predictions = predictions.reshape(-1, OUTPUT_LABEL_SIZE)
    df = DataFrame(predictions)
    df.to_excel('predictions.xlsx', header=None, index=None)