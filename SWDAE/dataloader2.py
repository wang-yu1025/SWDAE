import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
from torchvision import datasets, transforms


class MNIST_loader(data.Dataset):
    """Preprocessing을 포함한 dataloader를 구성"""

    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = x.resize((32,32),Image.ANTIALIAS)
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)




def get_mnist(args, data_dir='/home/htu/workspace/ZEJ/xb_svdd1one/data'):
    """get dataloders"""
    # min, max values for each class after applying GCN (as the original implementation)
    min_max = [(-0.8826567065619495, 9.001545489292527),
               (-0.6661464580883915, 20.108062262467364),
               (-0.7820454743183202, 11.665100841080346),
               (-0.7645772083211267, 12.895051191467457),
               (-0.7253923114302238, 12.683235701611533),
               (-0.7698501867861425, 13.103278415430502),
               (-0.778418217980696, 10.457837397569108),
               (-0.7129780970522351, 12.057777597673047),
               (-0.8280402650205075, 10.581538445782988),
               (-0.7369959242164307, 10.697039838804978)]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[args.normal_class][0]],
                                                         [min_max[args.normal_class][1] \
                                                          - min_max[args.normal_class][0]])])
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)
    
    x_train = train.data
    y_train = train.targets

    x_train = x_train[np.where(y_train == args.normal_class)]
    y_train = y_train[np.where(y_train == args.normal_class)]
    # x_train = x_train[:5120,...]
    # y_train = y_train[:5120]
    x_train = x_train[:5120,...]
    y_train = y_train[:5120]
    data_train = MNIST_loader(x_train, y_train, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
   
    x_test = test.data
    y_test = test.targets
    #x_test = x_test[np.where((y_test==args.normal_class )|(y_test==8))]
    #y_test = y_test[np.where((y_test==args.normal_class )|(y_test==8))]
    #x_test = x_test[:896,...]
    #y_test = y_test[:896]
    # Normal class인 경우 0으로 바꾸고, 나머지는 1로 변환 (정상 vs 비정상 class)
    y_test = np.where(y_test == args.normal_class, 0, 1)

    data_test = MNIST_loader(x_test, y_test, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x
