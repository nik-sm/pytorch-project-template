from torch import nn

from pytorch_project_template.models.base import BaseModel


class Classifier(BaseModel):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1, bias=False),  # Bias directly before BN is pointless
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 10),
            nn.LogSoftmax(-1),
        )

    def forward(self, data):
        return self.layers(data)
