import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet18
from utils import plot_history
import config
from tqdm import tqdm
import random
import numpy as np
from dataset import get_train_loaders
import os
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 设置python的哈希种子，python字典和其他哈希表结构依赖于哈希函数，而哈希函数的行为在不同运行之间可能会不同，通过设置环境变量，可使哈希结果在同一种子下保持一致；
    os.environ['PYTHONHASHSEED'] = str(seed)
    # tf.random.set_seed(seed)
    if torch.cuda.is_available():
        # 为当前GPU设置随机种子
        torch.cuda.manual_seed(seed)
        # 为所有可用的GPU设置相同的随机种子
        torch.cuda.manual_seed_all(seed)

        # https://docs.pytorch.ac.cn/docs/stable/notes/randomness.html
        # 使用场景：需要严格可重复性的实验。
        # cuDnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。
        # 用于保证CUDA卷积运算的结果确定，但是会让训练变得非常慢。
        # torch.backends.cudnn.deterministic = True

        # 禁用cuDNN的自动选择最佳卷积算法的功能，默认情况下cuDNN会在首次运行时尝试找到最适合硬件的算法，这可能会导致结果的不确定性，禁用此选项可以确保每次都是用相同的算法
        # torch.backends.cudnn.benchmark = False 

        # 仍需保证cuDNN是启用的？
        torch.backends.cudnn.enabled = True


def train():        
    # 预训练好的
    if args.model == 'resnet':
        model = resnet18(num_classes=config.num_classes)
    else:
        model = mobilenet_v2(num_classes=config.num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 优化器可以自己选择
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # 学习率的调度器也可以自己选择，或者删掉不用
    # warmup_epochs = 5
    # scheduler = optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[
    #         optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
    #         optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-warmup_epochs, eta_min=1e-5),
    #     ],
    #     milestones=[warmup_epochs]
    # )

    train_loader, val_loader = get_train_loaders(args.data_dir, args.batch_size, config.input_size, args.num_workers)

    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        with tqdm(train_loader, desc=f'Epoch({epoch}/{args.epochs}) [train]', unit='batch', leave=False) as train_pbar:
            for images, labels in train_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()


                current_train_loss = train_loss / (train_pbar.n + 1)
                current_train_acc = 100. * train_correct / train_total
                
                train_pbar.set_postfix({
                    'Loss': f'{current_train_loss:.4f}',
                    'Acc': f'{current_train_acc:.2f}%',
                })



        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)




        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch({epoch}/{args.epochs}) [valid]', unit='batch', leave=False) as val_pbar:
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                    current_val_loss = val_loss / (val_pbar.n + 1)
                    current_val_acc = 100. * val_correct / val_total

                    val_pbar.set_postfix({
                        'Loss': f'{current_val_loss:.4f}',
                        'Acc': f'{current_val_acc:.2f}%',
                    })

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        
        # 更新学习率
        # scheduler.step()


        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': history,
                },
                os.path.join(save_path, 'best_model.pth')
            )
            tqdm.write(f'New best model saved! val_acc: {val_acc:.2f}%')

    tqdm.write(f'Training completed! best_val_acc: {best_val_acc:.2f}%')
    tqdm.write(f'Final metrics - train_acc: {history["train_acc"][-1]:.2f}%, 'f'val_acc: {history["val_acc"][-1]:.2f}%')

    return history

# 可以用命令行配置参数，也可以在config.py里面配置
def parse_args():
    parser = argparse.ArgumentParser(description='Train the net on ASL images')
    
    # 数据集配置
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='Directory path for data')
    parser.add_argument('--num_workers', type=int, default=config.num_workers, help='Number of data loading workers')

    # 训练配置
    parser.add_argument('--seed', type=int, default=config.seed, help='Random seed')
    parser.add_argument('--epochs', '-e', type=int, default=config.epochs, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=config.learning_rate, help='Learning rate')

    parser.add_argument('--model', type=str, default=config.model, help='Model architecture (resnet18, mobilenet_v2, etc.)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = f'./runs/{time.strftime("%Y%m%d%H%M%S", time.localtime())}-{args.model}-bs{args.batch_size}-lr{args.learning_rate}-ep{args.epochs}'
    os.makedirs(save_path, exist_ok=True)

    history = train()

    plot_history(**history, save_path=save_path)