

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def filter_input(x):
  x[x<0] = 0  


def weighted_mse(input, target, weight, device):
  input = input.to(device)
  target = target.to(device)
  weight = weight.to(device)
  return torch.sum(weight * (input - target)**2)

def weighted_MSE(a, b, d, t, label_a, label_b, label_d, label_t, weight, device):
  a = a.to(device)
  b = b.to(device)
  d = d.to(device)
  t = t.to(device)

  label_a = label_a.to(device)
  label_b = label_b.to(device)
  label_d = label_d.to(device)
  label_t = label_t.to(device)

  weight = weight.to(device)
  
  a = a.view(len(a))
  b = b.view(len(b))
  d = d.view(len(d))
  t = t.view(len(t))

  crit1 = nn.MSELoss()
  crit2 = nn.MSELoss()
  crit3 = nn.MSELoss()
  crit4 = nn.MSELoss()

  return weight[0] * crit1(a, label_a) + weight[1] * crit2(b, label_b) + weight[2] * crit3(d, label_d) + weight[3] * crit4(t, label_t)

def save_plot(train_loss, valid_loss):
	# visualize the loss as the network trained
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
	plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

	# find position of lowest validation loss
	minposs = valid_loss.index(min(valid_loss))+1 
	plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, 300) # consistent scale
	plt.xlim(0, len(train_loss)+1) # consistent scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()
	fig.savefig('loss_plot.png', bbox_inches='tight')