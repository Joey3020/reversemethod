

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

def WMSE4(a_, b_, d_, t_, a, b, d, t, ratio, weight, device):
  a, b, d, t = a.to(device), b.to(device), d.to(device), t.to(device)
  a_, b_, d_, t_ = a_.to(device), b_.to(device), d_.to(device), t_.to(device)
  weight = weight.to(device)

  a_, b_, d_, t_ = a_.squeeze(), b_.squeeze, d_.squeeze(), t_.squeeze()

  loss_fn = nn.MSELoss()
  loss = weight[0] * loss_fn(ratio * a, a_) + weight[1] * loss_fn(ratio * b, b_) + weight[2] * loss_fn(d, d_) + weight[3] * loss_fn(t, t_)
  return loss

def WMSE3(a_, b_, d_, t_, a, b, d, t, weight, device):
  a, b, d, t = a.to(device), b.to(device), d.to(device), t.to(device)
  a_, b_, d_, t_ = a_.to(device), b_.to(device), d_.to(device), t_.to(device)
  weight = weight.to(device)

  a_, b_, d_, t_ = a_.squeeze(), b_.squeeze(), d_.squeeze(), t_.squeeze()

  loss_fn = nn.MSELoss()
  loss = weight[0] * loss_fn(a, a_) + weight[1] * loss_fn(b, b_) + weight[2] * loss_fn(d, d_) + weight[3] * loss_fn(t, t_)
  return loss

def WMSE2(a_, b_, d_, a, b, d, weight, device):
  a = a.to(device)
  b = b.to(device)
  d = d.to(device)
  a_ = a_.to(device)
  b_ = b_.to(device)
  d_ = d_.to(device)
  weight = weight.to(device)

  a = a.view(len(a))
  b = b.view(len(b))
  d = d.view(len(d))

  loss_fn = nn.MSELoss()

  loss = weight[0] * loss_fn(a, a_) + weight[1] * loss_fn(b, b_) + weight[2] * loss_fn(d, d_)
  return loss

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

def save_plot(train_loss, valid_loss, limit):
	# visualize the loss as the network trained
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
	plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

	# find position of lowest validation loss
	minposs = valid_loss.index(min(valid_loss))+1 
	plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, limit) # consistent scale
	plt.xlim(0, len(train_loss)+1) # consistent scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()
	fig.savefig('loss_plot.png', bbox_inches='tight')