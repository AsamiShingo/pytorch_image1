from matplotlib.pyplot import xcorr
import torch.nn as nn
import torch

class ImageNet(nn.Module):    
    def __init__(self, chanel, width, height, output_size):
        super().__init__()
        self.chanel = chanel
        self.width = width
        self.height = height
        self.layers = []
        
        c = self.chanel
        w = self.width
        h = self.height
        
        def create_conv_layer(in_c, out_c, w, h, is_batchnormal, is_maxpool, is_dropout):
            layers = []
            c = in_c
            
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
            c = out_c
            
            if is_batchnormal == True:
                layers.append(nn.BatchNorm2d(c))
            
            layers.append(nn.ReLU(inplace=False))
            
            if is_maxpool == True:
                layers.append(nn.MaxPool2d(2, 2))
                w = w // 2                    
                h = h // 2
                
            if is_dropout == True:
                layers.append(nn.Dropout(0.5, inplace=False))
                
            return layers, c, w, h
        
        layers_tmp, c, w, h = create_conv_layer(c, 64, w, h, False, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 64, w, h, True, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 64, w, h, False, True, True)
        self.layers += layers_tmp
        
        layers_tmp, c, w, h = create_conv_layer(c, 128, w, h, False, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 128, w, h, True, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 128, w, h, False, True, True)
        self.layers += layers_tmp
        
        layers_tmp, c, w, h = create_conv_layer(c, 256, w, h, False, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 256, w, h, True, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 128, w, h, False, False, False)
        self.layers += layers_tmp        
        
        layers_tmp, c, w, h = create_conv_layer(c, 256, w, h, False, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 256, w, h, True, False, False)
        self.layers += layers_tmp
        
        layers_tmp, c, w, h = create_conv_layer(c, 256, w, h, False, False, False)
        self.layers += layers_tmp
        layers_tmp, c, w, h = create_conv_layer(c, 256, w, h, True, False, False)
        self.layers += layers_tmp
        
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(c*w*h, 1024))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Dropout(0.5, inplace=False))
        self.layers.append(nn.Linear(1024, 1024))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Dropout(0.5, inplace=False))
        self.layers.append(nn.Linear(1024, output_size))
        self.layers.append(nn.Softmax(dim=1))
         
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, input):
        x = input
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def save_weight(self, path):
        torch.save(super().state_dict(), path)
        
    def load_weight(self, path):
        weights = torch.load(path)
        super().load_state_dict(weights)
        
        
if __name__=="__main__":
    imagenet = ImageNet(3, 32, 32)
    print(imagenet)