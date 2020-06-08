
from EarlyStopper import *
from utils import *
import neptune
import torch
import torch.optim as optim
from tqdm import trange
import os
import pandas
from pandas import DataFrame
import time

#train the model
def train(net, trainloader, valloader, weight, EPOCHS, LEARNING_RATE, IMG_CHANNELS, IMG_SIZE, MODEL_NAME, patience, validate_every, device):
    
    optimizer = optim.Adam(net.parameters(), LEARNING_RATE, betas=(0.9, 0.999), eps=1e-09, weight_decay=0, amsgrad=False)

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

                    x, y = x.to(device), y.to(device)

                    d, A, B, t = y[:, 2], y[:, 3], y[:, 4], y[:, 5]
                    d_, A_, B_, t_ = net(x)
                    
                    loss = WMSE3(d_, A_, B_, t_, d, A, B, t, weight, device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                net.eval()
                for batch_idx, samples in enumerate(valloader):
                    x, y = samples

                    filter_input(x)
                    x = x.view(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

                    x, y = x.to(device), y.to(device)

                    d, A, B, t = y[:, 2], y[:, 3], y[:, 4], y[:, 5]
                    d_, A_, B_, t_ = net(x)
                    
                    loss = WMSE3(d_, A_, B_, t_, d, A, B, t, weight, device)
                    val_losses.append(loss.item())

                train_loss = np.average(train_losses)
                valid_loss = np.average(val_losses)
                avg_train_losses.append(train_loss)
                avg_val_losses.append(valid_loss)
                train_losses = []
                val_losses = []

                neptune.log_metric('train loss', train_loss)
                neptune.log_metric('validation loss', valid_loss)

                f.write(f"{MODEL_NAME},{round(time.time(), 3)},  {round(float(train_loss), 4)},  {round(float(valid_loss),4)}\n")
                print("\nloss : ", train_loss, "val loss : ", valid_loss, "\n")
                
                early_stopping(valid_loss, net)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        net.load_state_dict(torch.load('checkpoint.pt'))
        neptune.log_artifact('checkpoint.pt')

    return avg_train_losses, avg_val_losses

def test(net, testloader, IMG_CHANNELS, IMG_SIZE, OUTPUT_LABEL_SIZE, device):
    predictions = []
    with torch.no_grad():
        for i, sample in enumerate(testloader):
            predict = []
            x ,y = sample

            filter_input(x)
            x = x.view(-1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
            
            x, y = x.to(device), y.to(device)
            
            d_, A_, B_, t_ = net(x)

            final_result = torch.cat([d_, A_, B_, t_], dim = 1)
            final_result = final_result.to("cpu")
          
            predictions.append(final_result.numpy())

    print(predictions)
    
    predictions = np.array(predictions)
    predictions = predictions.reshape(-1, OUTPUT_LABEL_SIZE)
    df = DataFrame(predictions)
    df.to_excel('predictions.xlsx', header=None, index=None)
    neptune.log_artifact('predictions.xlsx')