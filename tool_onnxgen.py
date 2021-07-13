import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os
from pathlib import Path
import shutil

if os.path.isdir('./output/net/'):
    print("Session already exists (./output/net/), overwrite the session? (y/n): ", end='')
    force_write = input()
    print("")
    if force_write == "y":
        try:
            shutil.rmtree("./output/net/")
            Path("./output/net/").mkdir(parents=True, exist_ok=True)
        except OSError:
            print("Error in session creation (./output/net/).")
            exit()
else:
    try:
        Path("./output/net/").mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error in session creation (./output/net/).")
        exit()



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 18, (1, 7))
        self.conv2 = nn.Conv2d(18, 18, (1, 7))

        self.pool = nn.MaxPool2d((1, 2))

        self.fc1 = nn.Linear(18 * 45, 100)
        self.fc2 = nn.Linear(100, 5)

#        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
#       x = x.view(-1,)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

dummy_input = torch.randn(1, 1, 1, 198)

torch.onnx.export(net,dummy_input,"output/net/net.onnx")
