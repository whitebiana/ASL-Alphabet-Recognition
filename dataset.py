import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch

__image_net_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
# __stats = {'mean': [0.4975, 0.2698, 0.0429], 'std': [0.2290, 0.2240, 0.2250]} # batch_size=16
# __stats = {'mean': [0.4975, 0.2698, 0.0429], 'std': [0.2290, 0.2240, 0.2250]} # batch_size=32
__stats = {'mean': [0.4975, 0.2698, 0.0429], 'std': [0.2290, 0.2240, 0.2250]} # batch_size=64

# 没用可以删掉
class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        self.labels = []

        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for img_name in os.listdir(label_dir):          
                img_path = os.path.join(label_dir, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(img_path)
                    self.labels.append(label)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.images)

# 没用可以删掉
def get_stats(loader):
    data_sum, data_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        data.cuda()
        # data: [batch_size, channels, height, width]
        # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
        data_sum += torch.mean(data, dim=[0, 2, 3])
        # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
        data_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        # 统计batch的数量
        num_batches += 1
    # 计算均值
    mean = data_sum / num_batches
    # 计算标准差
    std = (data_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean,std


def get_transforms(input_size):
    
    train_transform = v2.Compose([
        v2.RandomResizedCrop((input_size, input_size)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(30),
        v2.RandomPerspective(distortion_scale=0.2),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**__image_net_stats)
    ])

    val_transform = v2.Compose([
        v2.Resize((input_size, input_size)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**__image_net_stats)
    ])

    return train_transform, val_transform

# 没用可以删掉
def get_transform(input_size):
    return v2.Compose([
        v2.RandomResizedCrop((input_size, input_size)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(60),
        v2.RandomPerspective(distortion_scale=0.2),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**__image_net_stats)
    ])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_loaders(data_dir, batch_size, input_size, num_workers):
    train_transform, val_transform = get_transforms(input_size)
    # transform = get_transform(input_size)

    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, pin_memory=True)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)


    return train_loader, val_loader

def get_test_loader(data_dir, batch_size, input_size, num_workers):
    _, test_transform = get_transforms(input_size)

    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), test_transform)

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, pin_memory=True)

    return test_loader

# 没用可以删掉
if __name__ == "__main__":
    train_loader, _ = get_train_loaders()
    mean, std = get_stats(train_loader)
    print("Mean:", mean)
    print("Std:", std)