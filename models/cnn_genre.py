import torch
from torch import nn

class AudioCNN_0(nn.Module):
    def __init__(self, input_shape=1, hidden_units=16, output_shape=10):
        super().__init__()
        self.conv2d_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)          
        )

        self.conv2d_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 2,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 2),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.conv2d_block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2,
                      out_channels=hidden_units * 4,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 4),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 4 * 2 * 15,
                      out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=output_shape)                    
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.conv2d_block1(x)
        # print(x.shape)
        # x = self.conv2d_block2(x)
        # print(x.shape)
        # x = self.conv2d_block3(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv2d_block3(self.conv2d_block2(self.conv2d_block1(x))))
    

class AudioCNN_1(nn.Module):
    def __init__(self, input_shape=1, hidden_units=16, output_shape=10):
        super().__init__()
        self.conv2d_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)          
        )

        self.conv2d_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 2,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 2),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.conv2d_block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2,
                      out_channels=hidden_units * 4,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 4),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 4 * 2 * 15,
                      out_features=output_shape)                 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.conv2d_block1(x)
        # print(x.shape)
        # x = self.conv2d_block2(x)
        # print(x.shape)
        # x = self.conv2d_block3(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv2d_block3(self.conv2d_block2(self.conv2d_block1(x))))


class AudioCNN_2(nn.Module):
    def __init__(self, input_shape=1, hidden_units=16, output_shape=10):
        super().__init__()
        self.conv2d_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv2d_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 2,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 2),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv2d_block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2,
                      out_channels=hidden_units * 4,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 4),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        
        self.conv2d_block4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 4,
                      out_channels=hidden_units * 8,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 8),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 8 * 2 * 12,
                      out_features=output_shape)                 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.conv2d_block1(x)
        # print(x.shape)
        # x = self.conv2d_block2(x)
        # print(x.shape)
        # x = self.conv2d_block3(x)
        # print(x.shape)
        # x = self.conv2d_block4(x)
        # print(x.shape)
        # # x = self.global_pool(x)
        # # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv2d_block4(self.conv2d_block3(self.conv2d_block2(self.conv2d_block1(x)))))
    


class AudioCNN_3(nn.Module):
    def __init__(self, input_shape=1, hidden_units=16, output_shape=10):
        super().__init__()
        self.conv2d_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),         
            nn.BatchNorm2d(hidden_units),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.4)
        )

        self.conv2d_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 2,
                      kernel_size=3,
                      stride=1,
                      padding=0),        
            nn.BatchNorm2d(hidden_units * 2),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.4)
        )

        self.conv2d_block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2,
                      out_channels=hidden_units * 4,
                      kernel_size=3,
                      stride=1,
                      padding=0),       
            nn.BatchNorm2d(hidden_units * 4),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.4)
        )
        
        self.conv2d_block4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 4,
                      out_channels=hidden_units * 8,
                      kernel_size=3,
                      stride=1,
                      padding=0),         
            nn.BatchNorm2d(hidden_units * 8),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.4)
        )

        self.conv2d_block5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 8,
                      out_channels=hidden_units * 16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 16),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.5)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((5,5))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=hidden_units * 16 * 5 * 5,
                      out_features=output_shape)                 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Original: ",x.shape)
        # x = self.conv2d_block1(x)
        # print("Conv1: ", x.shape)
        # x = self.conv2d_block2(x)
        # print("Conv2: ", x.shape)
        # x = self.conv2d_block3(x)
        # print("Conv3: ", x.shape)
        # x = self.conv2d_block4(x)
        # print("Conv4: ", x.shape)
        # x = self.conv2d_block5(x)
        # print("Conv5: ", x.shape)
        # x = self.global_pool(x)
        # print("Glob: ", x.shape)
        # x = self.classifier(x)
        # print("Classifier: ", x.shape)
        # return x
        return self.classifier(self.global_pool(self.conv2d_block5(self.conv2d_block4(self.conv2d_block3(self.conv2d_block2(self.conv2d_block1(x)))))))
