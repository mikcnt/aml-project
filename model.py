import torch.nn as nn


class ColorizationNet(nn.Module):
    def __init__(self, type='classification'):
        super(ColorizationNet, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1),
        )

        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1),
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1),
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
        )

        self.model8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
        )

        if type == 'classification':
            self.model9 = nn.Sequential(
                nn.Conv2d(128, 313, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
                nn.Upsample(scale_factor=4),
                nn.Softmax(dim=1)
            )
        elif type == 'regression':
            self.model9 = nn.Sequential(
                nn.Conv2d(128, 2, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
                nn.Upsample(scale_factor=4),
                nn.Softmax(dim=1)
            )

    def forward(self, input_image):
        x = self.model1(input_image)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.model6(x)
        x = self.model7(x)
        x = self.model8(x)
        return self.model9(x)
