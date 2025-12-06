# 数据集配置
data_dir = "Synthetic_ASL"
num_classes = 27
num_workers = 8
input_size = 224

# 训练配置
seed = 1
epochs = 30
batch_size = 64
learning_rate = 0.05

# 模型配置
model = "resnet18"
# model = "mobilenet_v2"