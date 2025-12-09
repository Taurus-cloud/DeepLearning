import torch
from torch.utils.data import TensorDataset  # 数据集对象  数据->Tensor->数据集->数据加载器
from torch.utils.data import DataLoader     # 数据加载器
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary            # 模型参数计算

# 1.定义函数，构建数据集
def create_dataset():
    # 1.1 读取数据
    data = pd.read_csv('./手机价格预测.csv')

    # 1.2 数据预处理
    x, y = data.iloc[:, :-1], data.iloc[:,-1]
    x = x.astype(np.float32)
    # 1.3 划分训练集和测试集
    # 参1：特征数据 参2：标签数据 参3：测试集比例 参4：随机种子 参5：分层抽样
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)
    # 1.4 转换为Tensor
    x_train_tensor = torch.tensor(x_train.values)
    y_train_tensor = torch.tensor(y_train.values)
    x_test_tensor = torch.tensor(x_test.values)
    y_test_tensor = torch.tensor(y_test.values)
    # 1.5 创建数据集对象
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    # 1.6 返回数据集对象，特征维度，类别数
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))
# 2.搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim): # 输入特征数20，输出类别数4
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.output(x)
        return x
# 3.模型训练
def train(train_dataset,input_dim,output_dim):
    # 3.1 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 3.2 创建模型对象
    model = PhonePriceModel(input_dim, output_dim)
    # 3.3 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 3.4 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 3.5 训练过程
    epochs = 50
    for epoch in range(epochs):
        # 定义变量 记录每次训练损失值，训练批次数
        total_loss,batch_num=0.0,0
        # 记录训练开始时间
        start = time.time()
        for x,y in train_loader:
            model.train() # 设置模型为训练模式
            y_pred = model(x) # 前向传播
            loss = criterion(y_pred, y) # 计算损失
            optimizer.zero_grad() # 清空梯度
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print(f'Epoch:{epoch+1}, Loss:{total_loss /batch_num:.4}, Time:{time.time()-start:.2f}s')
    # 保存模型参数
    torch.save(model.state_dict(), 'phone_price_model.pth')
    print('Training complete')
# 4.模型测试
def evaluate(test_dataset,input_dim,output_dim):
    # 4.1 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 4.2 创建模型对象
    model = PhonePriceModel(input_dim, output_dim)
    # 4.3 加载模型参数
    model.load_state_dict(torch.load('phone_price_model.pth'))
    model.eval()  # 设置模型为评估模式
    # 4.5 测试过程
    correct = 0
    for x,y in test_loader:
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)
        correct += (y_pred == y).sum()
    accuracy = correct / len(test_dataset)
    print(f"准确率: {accuracy:.4f}")


# 5.测试
if __name__ == '__main__':
    # 1.准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f'训练集 数据集对象:{train_dataset}')
    # print(f'测试集 数据集对象:{test_dataset}')
    # print(f'输入特征数:{input_dim}')
    # print(f'输出标签数:{output_dim}')

    # # 2.构建神经网络模型
    # model = PhonePriceModel(input_dim,output_dim)
    # # 计算模型参数
    # # 参1：模型对象 参2：输入数据的形状(每批16条，每条数据input_dim个特征)
    # summary(model,input_size=(16,input_dim))
    
    # # 3.模型训练
    # train(train_dataset,input_dim,output_dim)

    # 4.模型测试
    evaluate(test_dataset,input_dim,output_dim)