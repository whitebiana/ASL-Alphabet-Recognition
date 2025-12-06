import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import mobilenet_v2, resnet18
import config
from dataset import get_test_loader
import argparse
import cv2
from utils import plot_confusion_matrix
from tqdm import tqdm

def test():

    # 加载训练好的模型参数
    if config.model == 'resnet18':
        model = resnet18(num_classes=config.num_classes).to(device)
        checkpoint = torch.load('./weights/resnet18/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = mobilenet_v2(num_classes=config.num_classes).to(device)
        checkpoint = torch.load('./weights/mobilenet_v2/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = get_test_loader(config.data_dir, config.batch_size, config.input_size, config.num_workers)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(test_loader, desc='[Test]', unit='batch') as test_pbar:
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    plot_confusion_matrix(confusion_matrix(all_labels, all_preds), class_names)

# 没用可以删掉
# def parse_args():
#     parser = argparse.ArgumentParser(description='Test the net on ASL images')
    
#     parser.add_argument('--data_dir', type=str, default=config.data_dir, help='Directory path for data')
#     parser.add_argument('--num_classes', '-c', type=int, default=config.num_classes, help='Number of classes')
#     parser.add_argument('--num_workers', type=int, default=config.num_workers, help='Number of data loading workers')
#     parser.add_argument('--input_size', type=int, default=config.input_size, help='Input image size')
#     parser.add_argument('--batch-size', '-b', type=int, default=config.batch_size, help='Batch size')
#     parser.add_argument('--weight', type=str, default=False, help='Load model from a .pth file')

#     args = parser.parse_args()
#     return args

if __name__ == '__main__':
    # args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = os.listdir(os.path.join(config.data_dir, 'test'))

    test()