import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import TeacherVAE
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 檢查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print(f"使用GPU 1: {torch.cuda.get_device_name(1)}")
else:
    device = torch.device("cpu") 
    print("無法使用GPU,使用CPU替代")

# 設定超參數
batch_size = 128
epochs = 50
learning_rate = 1e-3
latent_dim = 2

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 載入 MNIST 資料集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = TeacherVAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定義損失函數
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# 訓練函數
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)

# 測試函數
def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var).item()
    
    test_loss /= len(test_loader.dataset)
    return test_loss

# 訓練模型
train_losses = []
test_losses = []

print("開始訓練教師VAE模型...")
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss = test()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    print(f'Epoch: {epoch}')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}\n')
    
    # 每10個epoch儲存一些重建的圖像
    if epoch % 10 == 0:
        with torch.no_grad():
            # 從測試集取8張圖片
            data = next(iter(test_loader))[0][:8].to(device)
            recon_batch, _, _ = model(data)
            
            # 繪製原始圖片和重建圖片的比較
            plt.figure(figsize=(16, 4))
            for i in range(8):
                # 原始圖片
                plt.subplot(2, 8, i + 1)
                plt.imshow(data[i][0].cpu().numpy(), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Original')
                
                # 重建圖片
                plt.subplot(2, 8, i + 9)
                plt.imshow(recon_batch[i].view(28, 28).cpu().numpy(), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Reconstructed')
            
            plt.tight_layout()
            plt.savefig(f'figures/teacher_vae_reconstruction_epoch_{epoch}.png')
            plt.close()

# 儲存模型
torch.save(model.state_dict(), 'teacher_vae_model.pth')

# 繪製訓練結果
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Teacher VAE Training Progress')
plt.tight_layout()
plt.savefig('figures/teacher_vae_training_loss.png')
plt.close()

# 生成一些隨機樣本
with torch.no_grad():
    # 從標準正態分佈採樣
    sample = torch.randn(64, latent_dim).to(device)
    # 解碼樣本
    sample = model.decode(sample).cpu()
    
    # 繪製生成的圖片
    plt.figure(figsize=(8, 8))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(sample[i].view(28, 28).numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figures/teacher_vae_samples.png')
    plt.close() 