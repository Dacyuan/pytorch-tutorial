import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#超参数
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

#创建线性回归模型
model = nn.Linear(input_size, output_size)

#损失函数和优化函数

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#训练模型
for epoch in range(num_epochs):

    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    #前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    #后向传播 和 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print("Epoch[{}/{}],loss:{:.4f}".format(epoch+1, num_epochs, loss.item()))

#可视化
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train,"ro", label="Origin data")
plt.plot(x_train, predicted, label="Fitted line")
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')