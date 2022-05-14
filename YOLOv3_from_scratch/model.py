from statistics import mode
import torch
import torch.nn as nn
from torchsummary import summary
# tuple:(out_channels, kernal_size, stride)
# List:['B':num_repeats] 'U' upsampling 'S': sclae prediction
config  = [
    (32,3,1),# [32, 416, 416]
    (64,3,2),# [64, 208, 208]
    ['B',1], # [64, 208, 208]
    (128,3,2),# [128, 104, 104]
    ['B',2],# [128, 104, 104]
    (256,3,2), # [256, 52, 52]
    ['B',8], # add1[256, 52, 52]
    (512,3,2), # [512, 26, 26]
    ['B',8], # add2      [512, 26, 26]
    (1024,3,2),# [1024, 13, 13]
    ['B',4], # To this point is Darknet-53 [1024, 13, 13]
    (512,1,1), # [512, 13, 13]
    (1024,3,1),# !![1024, 13, 13]
    'S', # [1024, 13, 13]-> [512, 13, 13](continue)->[1024, 13, 13]->[3, 13, 13, 85] 其余scale prediction层道理相同
    (256,1,1),# [256, 13, 13]
    'U', # [256, 26, 26] - > + add2 [768, 26, 26]  
    (256,1,1),# [256, 26, 26]
    (512,3,1),# !![512, 26, 26]
    'S',# [3, 26, 26, 85]
    (128,1,1),# [128, 26, 26]
    'U',# [128, 52, 52] -> +add1 [384, 52, 52]
    (128,1,1),# [128, 52, 52]
    (256,3,1),# !![256, 52, 52]
    'S',# [3, 52, 52, 85]
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act= True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels, bias= not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

        
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual = True, num_repeats = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size = 1),
                    CNNBlock(channels // 2, channels , kernel_size = 3, padding = 1),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size = 3, padding =1),
            # no bn and relu bn_act = False
            CNNBlock(2*in_channels, 3*(num_classes + 5), bn_act=False, kernel_size = 1), #3* [p0, x,y,w,h]
        )
        self.num_classes = num_classes
    def forward(self, x):
        # x = self.pred(x)
        return(
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0,1,3,4,2)
            #[batch_size, 3,  self.num_classes + 5, 13/26/52, 13/26/52]
        )
    

class YOLOv3(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 80, ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        # multi-output, keep track
        outputs = []
        # show us where to concatenate the channels
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                # print(x.shape)
                outputs.append(layer(x))
                # save the output and go back
                continue
            # print(layer)
            # print(x.shape)
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
                
            elif isinstance(layer, nn.Upsample):
                # print(len(route_connections))
                x = torch.cat([x, route_connections[-1]], dim = 1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels= self.in_channels
        for module in config:
            # print(in_channels)
            # if it's a tuple like (32,3,1),
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                    in_channels,
                    out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = 1 if kernel_size == 3 else 0,
                    )
                )
                # next in_channels
                in_channels = out_channels
            # like ['B',2]
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            # like 'S' 'U'
            elif isinstance(module, str):
                if module == 'S':
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size =1),
                        ScalePrediction(in_channels //2, num_classes= self.num_classes),
                    ]
                    in_channels = in_channels // 2
                elif module == 'U': 
                    layers.append(nn.Upsample(scale_factor=2))
                    # concatenate 以后通道数乘3 as we can see 
                    in_channels = in_channels * 3
        return layers
                
# unit test!
if __name__ == "__main__":
    num_classes = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_SIZE = 416 # Yolov1:448 Yolov3:416 
    model = YOLOv3(num_classes=num_classes).to(device)
    # model = YOLOv3(num_classes=num_classes)
    print(summary(model, (3,IMAGE_SIZE, IMAGE_SIZE)))
    # 1/0
    # x = torch.randn((32,3,IMAGE_SIZE, IMAGE_SIZE))
    # # 416 / 32 = 13
    # out = model(x)
    # # torch.Size([batch_size = 32, 3, 13, 13, 85])
    # # torch.Size([batch_size, 3, 26, 26, 85])
    # # torch.Size([batch_size, 3, 52, 52, 85])
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out[2].shape)

    
