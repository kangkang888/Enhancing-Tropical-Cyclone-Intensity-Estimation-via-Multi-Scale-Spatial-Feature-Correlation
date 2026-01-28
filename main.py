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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from model import *
import cv2
from scipy import ndimage
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(2021)

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

unique_storm_ids = data_info['ID'].unique()

# Handle Time column (lowercase in dataset) and Extract Year from ID
if 'time' in data_info.columns:
    # Format appears to be YYYYMMDDHH, e.g., 2017041606
    data_info['Time'] = pd.to_datetime(data_info['time'].astype(str), format='%Y%m%d%H')
else:
    # Fallback if 'time' contains standard datetime objects or strings
    data_info['Time'] = pd.to_datetime(data_info['time'])

# Extract Year from ID (as requested by user, first 4 chars)
# Assuming ID format like '201701L'
data_info['Year'] = data_info['ID'].astype(str).str[:4].astype(int)

# Sort storms strictly by Start Time
storm_start_times = data_info.groupby('ID')['Time'].min()
sorted_storm_ids = storm_start_times.sort_values().index.tolist()

# Determine split indices for 8:1:1 (Train: 80%, Val: 10%, Test: 10%)
n_storms = len(sorted_storm_ids)
split_idx1 = int(n_storms * 0.8)
split_idx2 = int(n_storms * 0.9)

train_storm_ids = sorted_storm_ids[:split_idx1]
val_storm_ids = sorted_storm_ids[split_idx1:split_idx2]
test_storm_ids = sorted_storm_ids[split_idx2:]

# Print split stats for verification
# Using extracted Year from ID for verification as requested
train_years = data_info[data_info['ID'].isin(train_storm_ids)]['Year']
val_years = data_info[data_info['ID'].isin(val_storm_ids)]['Year']
test_years = data_info[data_info['ID'].isin(test_storm_ids)]['Year']

print(f"Training Data Years: {train_years.min()} - {train_years.max()} (Count: {len(train_storm_ids)})")
print(f"Validation Data Years: {val_years.min()} - {val_years.max()} (Count: {len(val_storm_ids)})")
print(f"Testing Data Years: {test_years.min()} - {test_years.max()} (Count: {len(test_storm_ids)})")

train_mask = data_info['ID'].isin(train_storm_ids)
val_mask = data_info['ID'].isin(val_storm_ids)
test_mask = data_info['ID'].isin(test_storm_ids)

X_train = X_irpmw[train_mask]
y_train = y[train_mask]
info_train = data_info[train_mask].reset_index(drop=True)

X_val = X_irpmw[val_mask]
y_val = y[val_mask]
info_val = data_info[val_mask].reset_index(drop=True)

X_test = X_irpmw[test_mask]
y_test = y[test_mask]
info_test = data_info[test_mask].reset_index(drop=True) # crucial for post-processing smoothing

# Normalize inputs if needed (already handled by model/preprocessing usually, explicitly casting here)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
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
    #模型改进
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
    best_val_loss = float('inf') # Use Validation Loss for early stopping
    best_model_path = 'best_model.pth'
    
    # Try to load existing model
    try:
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model from", best_model_path)
    except FileNotFoundError:
        print("No best model found, starting from scratch")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
        
        # 验证模型 (Use Validation Set)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                inputs = inputs.permute(0, 3, 1, 2).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f'New Best Model saved with Val Loss: {best_val_loss:.4f}')
                
        train_losses.append(avg_train_loss)
        test_losses.append(avg_val_loss) # Logging Val loss as "test_loss" for plot compatibility, or change variable name
    with open('DCNSESP_threechannel_losses.txt', 'w') as f:
        for epoch in range(num_epochs):
            f.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[epoch]:.4f}, Val Loss: {test_losses[epoch]:.4f}\n')
    # 绘制训练损失和测试损失的变化
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    # plt.plot(test_losses, label='Validation Loss') # Removed as requested
    plt.title('Training Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 在显示图表前保存到本地
    plt.savefig('DCNSESP_threechannel_loss_plot.png', format='png', dpi=300)
    plt.show()

    print("Loading Best Model for Final Evaluation on Test Set...")
    try:
        model.load_state_dict(torch.load(best_model_path))
        print("Best model loaded successfully.")
    except FileNotFoundError:
        print("Warning: Best model file not found, using last epoch model.")

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

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    info_test['Predicted'] = all_preds
    info_test['Actual'] = all_targets
    
    # 计算RMSE和R (Pearson)
    final_preds = all_preds
    rmse = np.sqrt(mean_squared_error(all_targets, final_preds))
    # Pearson Correlation
    r_pearson, _ = pearsonr(all_targets, final_preds)

    # 绘制真实值与预测值的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, final_preds, alpha=0.5)
    plt.xlabel('True Intensity')
    plt.ylabel('Predicted Intensity (Smoothed)')
    plt.title(f'True vs Predicted Intensity\nRMSE: {rmse:.4f}, Pearson R: {r_pearson:.4f}')
    plt.grid(True)
    plt.savefig('DCNSESP_threechannel_true_vs_predicted_smoothed.png', format='png', dpi=300)
    plt.show()

    print(f'RMSE: {rmse:.4f}')
    print(f'Pearson R: {r_pearson:.4f}')
