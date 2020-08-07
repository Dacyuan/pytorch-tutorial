import torch
import torch.nn  as nn 
import numpy as np
import torchvision
import torchvision.transforms as transforms 

# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

#创建tensor
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

#构建计算图
y = w*x + b 

#计算梯度
y.backward()

#打印梯度
print(x.grad)
print(w.grad)
print(b.grad)




# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

#创建tensor 的形状为(10, 3) and (10, 2)

x = torch.randn(10, 3)
y = torch.randn(10, 2)

#构建一个全连接层
linear = nn.Linear(3, 2)
print("w:", linear.weight)
print("b:", linear.bias)

#创建损失函数 和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

#前向传播
pred = linear(x)

#计算损失
loss = criterion(pred, y)
print("loss:", loss.item())

#反向传播
loss.backward()

#打印梯度
print("dL/dw:", linear.weight.grad)
print("dL/db:", linear.bias.grad)


# 1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())




# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

#创建一个numpy数组
x = np.array([[1, 2], [3, 4]])

#将numpy数组转为tensor
y = torch.from_numpy(x)

#将tensor转为numpy
z = y.numpy()


# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #

#下载和构造数据集

train_dataset = torchvision.datasets.CIFAR10(root="../../data/",
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

#拿数据（从硬盘中读取数据）
image, label = train_dataset[0]
print(image.size())
print(label)

#加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

#Mini-batch images and labels
images, labels = data_iter.next()

#Actual usage of the data loader is as below
for images, labels in train_loader:
    #训练数据 写在这里
    pass


# ================================================================== #
#                5. Input pipeline for custom dataset                 #
# ================================================================== #

#构造自己定义的数据集
class CustomDataset(torch.utils.data.Dataset):

    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))