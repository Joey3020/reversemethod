import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__(self, INPUT_CHANNELS, OUTPUT1, OUTPUT2):
		super().__init__()

		self.INPUT_CHANNELS = INPUT_CHANNELS
		self.OUTPUT1 = OUTPUT1
		self.OUTPUT2 = OUTPUT2

		self.cnn = nn.Sequential(
			nn.Conv2d(self.INPUT_CHANNELS, 8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#32, 32, 8
			
			nn.Conv2d(8, 12, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#16, 16, 12
			
			nn.Conv2d(12, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#8, 8, 16
			
			nn.Conv2d(16, 20, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#4, 4, 20
			
			nn.Conv2d(20, 24, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)
			#2, 2, 24
		)
		self.fccAB = nn.Sequential(
			nn.Linear(96, 32),
			nn.ReLU(),
			nn.Linear(32, 10),
			nn.ReLU(),
			nn.Linear(10, self.OUTPUT1)
		)

		self.fccdt = nn.Sequential(
			nn.Linear(96, 32),
			nn.ReLU(),
			nn.Linear(32, 10),
			nn.ReLU(),
			nn.Linear(10, self.OUTPUT2)
		)

	def forward(self, x):

		feature = self.cnn(x)

		feature = feature.view(feature.size(0), -1)

		output1 = self.fccAB(feature)
		A = output1[:,:1]
		B = output1[:,1:2]

		output2 = self.fccdt(feature)
		d = output2[:, :1]
		t = output2[:, 1:2]

		return d, A, B, t
