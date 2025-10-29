# get the start time
import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from model import *
import cv2
from scipy import ndimage

data_path = "./data/TCIR-ATLN_EPAC_WPAC.h5"
data_info = pd.read_hdf(data_path, key="info", mode='r')
with h5py.File(data_path, 'r') as hf:
    data_matrix = hf['matrix'][:]

print(data_matrix.shape)

def smooth_three_channels(data, gaussian_sigma=1.0, bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75, median_kernel=3):
    """
    对三通道数据分别进行高斯滤波、双边滤波和中值滤波平滑处理
    
    Args:
        data: 输入数据，形状为 (N, H, W, C)，其中C=3表示三个通道
        gaussian_sigma: 高斯滤波的标准差
        bilateral_d: 双边滤波的邻域直径
        bilateral_sigma_color: 双边滤波的颜色空间标准差
        bilateral_sigma_space: 双边滤波的坐标空间标准差
        median_kernel: 中值滤波的核大小
    
    Returns:
        smoothed_data: 平滑处理后的数据
    """
    smoothed_data = np.zeros_like(data)
    
    for i in tqdm(range(data.shape[0]), desc="处理样本"):
        for c in range(data.shape[3]):  # 遍历三个通道
            channel_data = data[i, :, :, c].astype(np.float32)
            
            # 对每个通道应用不同的滤波方法
            if c == 0:  # 第一个通道（红外）使用高斯滤波
                smoothed_data[i, :, :, c] = ndimage.gaussian_filter(channel_data, sigma=gaussian_sigma)
            elif c == 1:  # 第二个通道（水汽）使用双边滤波
                # 双边滤波需要将数据转换为uint8格式
                channel_uint8 = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                bilateral_filtered = cv2.bilateralFilter(channel_uint8, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
                # 转换回原始数据范围
                smoothed_data[i, :, :, c] = bilateral_filtered.astype(np.float32) * (channel_data.max() - channel_data.min()) / 255.0 + channel_data.min()
            else:  # 第三个通道（被动微波降雨率）使用中值滤波
                smoothed_data[i, :, :, c] = ndimage.median_filter(channel_data, size=median_kernel)
    
    return smoothed_data

# keep only IR and PMW
#X_irpmw = data_matrix[:,:,:,0::3]
#三通道效果比两通道要好一点
X_irpmw = data_matrix[:,:,:,[0,1,3]]
y = data_info['Vmax'].values[:]

X_irpmw[np.isnan(X_irpmw)] = 0
X_irpmw[X_irpmw > 1000] = 0

# 对三通道数据进行平滑处理
X_irpmw = smooth_three_channels(X_irpmw)

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X_irpmw, y, test_size=0.1, random_state=42)
# Define the input shape
input_shape = (201, 201, 3)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt(self.MSE(x, y)) + self.eps

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    #稍加修改
    #model = ResNet18().to(device)

    #效果好renset-18
    #model = ResNetRegressor().to(device)


    #加入CBAM
    #model = ResNetRegressorCBAM().to(device)

    #CA
    #model = ResNetRegressorCA().to(device)

    #SE(8.92)效果最好
    #model = ResNetRegressorSE().to(device)

    #CBAMSE
    # model = ResNetRegressorCBAMSE().to(device)

    #CASE
    #model = ResNetRegressorCASE().to(device)

    #DCN(8.94)
    #model =ResNetRegressorDCN().to(device)

    #DCNSE
    # model = ResNetRegressorDCNSE().to(device)

    #DCNCBAMSE
    # model = ResNetRegressorDCNCBAMSE().to(device)

    #SP
    #model = ResNetRegressorSP().to(device)
    #DCNSESP
    model = ResNetRegressorDCNSESP().to(device)
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    train_losses = []
    test_losses = []

    # 训练模型
    num_epochs = 100
    #这里要手动改第二次要改成上次的最小损失
    best_test_loss = float('inf')
    best_model_path = 'best_model.pth'
    try:
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model from", best_model_path)
    except FileNotFoundError:
        print("No best model found, starting from scratch")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}')
        # 测试模型
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs, targets in tqdm(test_loader):
                inputs = inputs.permute(0, 3, 1, 2).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            print(f'Test Loss: {test_loss / len(test_loader):.4f}')
            if (test_loss / len(test_loader)) < best_test_loss:
                best_test_loss = test_loss / len(test_loader)
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved with test loss: {best_test_loss}')
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
    # 将损失写入文件
    with open('DCNSESP_threechannel_losses.txt', 'w') as f:
        for epoch in range(num_epochs):
            f.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[epoch]:.4f}, Test Loss: {test_losses[epoch]:.4f}\n')
    # 绘制训练损失和测试损失的变化
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.title('Training and Testing Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 在显示图表前保存到本地
    plt.savefig('DCNSESP_threechannel_loss_plot.png', format='png', dpi=300)
    plt.show()

    # 绘制真实值与预测值的散点图，并计算R值
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 计算RMSE和R
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = np.sqrt(r2_score(all_targets, all_preds))

    # 绘制真实值与预测值的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.xlabel('True Intensity')
    plt.ylabel('Predicted Intensity')
    plt.title(f'True vs Predicted Intensity\nRMSE: {rmse:.4f}, R: {r2:.4f}')
    plt.grid(True)
    plt.savefig('DCNSESP_threechannel_true_vs_predicted.png', format='png', dpi=300)
    plt.show()

    print(f'R: {r2:.4f}')
