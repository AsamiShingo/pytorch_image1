from imagedataset import ImageDataset, ImageDataLoader
from imagenet import ImageNet
from imagepytorch import ImagePytorch
import os
import sys

if __name__=="__main__":
    
    datapath=""
    if len(sys.argv) == 2:
        datapath=sys.argv[1]
    else:
        datapath=r"C:\work\asami\code\pytorch_image1\data\cifar10\cifar10_data"
           
    dataset_train = ImageDataset(True, 32)
    dataset_train.load_numpys(os.path.join(datapath, "train"))
    dataloader_train = ImageDataLoader(dataset_train, 50)
    
    dataset_test = ImageDataset(False, 32)
    dataset_test.load_numpys(os.path.join(datapath, "test"))
    dataloader_test = ImageDataLoader(dataset_test, 50)
        
    net = ImageNet(3, 32, 32, 10)
    image_train = ImagePytorch(net)
    
    weight_path=os.path.join(datapath, "testdata.dat")
    if os.path.isfile(weight_path):
        net.load_weight(weight_path)
        
    image_train.train(dataloader_train, dataloader_test, 2)
    
    net.save_weight(weight_path)
    