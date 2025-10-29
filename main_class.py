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


# 根据风速（vmax）进行分类的函数
def classify_wind_speed(vmax):
    if vmax <= 10.3:
        return 'NC'
    elif vmax <= 17.0:
        return 'TD'
    elif vmax <= 32.4:
        return 'TS'
    elif vmax <= 42.2:
        return 'H1'
    elif vmax <= 48.9:
        return 'H2'
    elif vmax <= 57.6:
        return 'H3'
    elif vmax <= 70.0:
        return 'H4'
    else:
        return 'H5'


data_path = "./data/TCIR-ATLN_EPAC_WPAC.h5"
data_info = pd.read_hdf(data_path, key="info", mode='r')
with h5py.File(data_path, 'r') as hf:
    data_matrix = hf['matrix'][:]

print(data_matrix.shape)

# 只保留IR和PMW
X_irpmw = data_matrix[:, :, :, [0, 1, 3]]
y = data_info['Vmax'].values[:]

X_irpmw[np.isnan(X_irpmw)] = 0
X_irpmw[X_irpmw > 1000] = 0

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X_irpmw, y, test_size=0.1, random_state=42)

# 定义输入形状
input_shape = (201, 201, 3)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt(self.MSE(x, y)) + self.eps


device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    # 稍加修改
    # model = ResNet18().to(device)

    # 效果好renset-18
    # model = ResNetRegressor().to(device)

    # 加入CBAM
    #model = ResNetRegressorCBAM().to(device)

    # CA
    model = ResNetRegressorCA().to(device)

    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    train_losses = []
    test_losses = []

    # 训练模型
    num_epochs = 200
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
        category_train_losses = {cat: [] for cat in ['NC', 'TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5']}

        for inputs, targets in tqdm(train_loader):
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 分类损失
            categories = [classify_wind_speed(v.item()) for v in targets]
            for cat, pred, target in zip(categories, outputs, targets):
                cat_loss = criterion(pred.view(1), target.view(1)).item()
                category_train_losses[cat].append(cat_loss)

        avg_train_losses = {cat: np.mean(losses) for cat, losses in category_train_losses.items()}
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}')
        for cat, avg_loss in avg_train_losses.items():
            print(f'Category {cat} Train Loss: {avg_loss:.4f}')

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            category_test_losses = {cat: [] for cat in ['NC', 'TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5']}

            for inputs, targets in tqdm(test_loader):
                inputs = inputs.permute(0, 3, 1, 2).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # 分类损失
                categories = [classify_wind_speed(v.item()) for v in targets]
                for cat, pred, target in zip(categories, outputs, targets):
                    cat_loss = criterion(pred.view(1), target.view(1)).item()
                    category_test_losses[cat].append(cat_loss)

            avg_test_losses = {cat: np.mean(losses) for cat, losses in category_test_losses.items()}
            print(f'Test Loss: {test_loss / len(test_loader):.4f}')
            if (test_loss / len(test_loader)) < best_test_loss:
                best_test_loss = test_loss / len(test_loader)
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved with test loss: {best_test_loss}')
            for cat, avg_loss in avg_test_losses.items():
                print(f'Category {cat} Test Loss: {avg_loss:.4f}')

        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        # 将损失写入文件
    with open('ca_threechannel_losses.txt', 'w') as f:
        for epoch in range(num_epochs):
            f.write(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[epoch]:.4f}, Test Loss: {test_losses[epoch]:.4f}\n')
    # 按类别计算测试损失
    model.eval()
    category_losses = {}
    all_preds = []
    all_targets = []
    all_categories = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_categories.append([classify_wind_speed(v) for v in targets.cpu().numpy()])

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_categories = np.concatenate(all_categories)

    unique_categories = np.unique(all_categories)

    for category in unique_categories:
        category_indices = np.where(all_categories == category)
        category_targets = all_targets[category_indices]
        category_preds = all_preds[category_indices]
        category_rmse = np.sqrt(mean_squared_error(category_targets, category_preds))
        category_losses[category] = category_rmse

    # 绘制每一类的损失
    plt.figure(figsize=(10, 6))
    categories = list(category_losses.keys())
    losses = list(category_losses.values())
    bars = plt.bar(categories, losses, color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('RMSE Loss')
    plt.title('RMSE Loss per Category')

    # 在每个柱子顶部添加损失值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 4), ha='center', va='bottom')

    plt.savefig('category_rmse_losses.png', format='png', dpi=300)
    plt.show()

    # 绘制训练损失和测试损失的变化
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.title('Training and Testing Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 在显示图表前保存到本地
    plt.savefig('ca_threechannel_loss_plot.png', format='png', dpi=300)
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
    plt.savefig('ca_threechannel_true_vs_predicted.png', format='png', dpi=300)
    plt.show()

    print(f'R: {r2:.4f}')
