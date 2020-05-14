import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__(self, INPUT_CHANNELS, OUTPUT_AB, OUTPUT_D_THETA):
		super().__init__()

		self.INPUT_CHANNELS = INPUT_CHANNELS
		self.OUTPUT_AB = OUTPUT_AB
		self.OUTPUT_D_THETA = OUTPUT_D_THETA

		self.cnn = nn.Sequential(
			nn.Conv2d(self.INPUT_CHANNELS, 12, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#32, 32, 12
			
			nn.Conv2d(12, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#16, 16, 16
			
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#8, 8, 32
			
			nn.Conv2d(32, 48, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			#4, 4, 48
			
			nn.Conv2d(48, 60, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)
			#2, 2, 60
		)
		self.fcc_ab = nn.Sequential(
			nn.Linear(240, 120),
			nn.ReLU(),
			nn.Linear(120, 30),
			nn.ReLU(),
			nn.Linear(30, self.OUTPUT_AB)
		)
		self.fcc_d_theta = nn.Sequential(
			nn.Linear(240, 120),
			nn.ReLU(),
			nn.Linear(120, 30),
			nn.ReLU(),
			nn.Linear(30, self.OUTPUT_D_THETA)
		)

	def forward(self, x):

		feature = self.cnn(x)

		feature = feature.view(feature.size(0), -1)

		ab = self.fcc_ab(feature)
		dt = self.fcc_d_theta(feature)

		a = ab[:, :1]
		b = ab[:, 1:]
		d = dt[:, :1]
		t = dt[:, 1:]

		return a, b, d, t
