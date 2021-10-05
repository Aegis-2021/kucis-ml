import torch

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        # 첫번째층
        # Input -> (?, 256, 256, 1)
        # Conv  -> (?, 256, 256, 32)
        # Pool  -> (?, 128, 128, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # Input -> (?, 128, 128, 32)
        # Conv  -> (?, 128, 128, 64)
        # Pool  -> (?, 64, 64, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 64x64x64 inputs -> 9 outputs
        self.fc = torch.nn.Linear(64 * 64 * 64, 9, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out
