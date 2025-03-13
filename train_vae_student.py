import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import TeacherVAE, StudentVAE
from tqdm import tqdm
import matplotlib.pyplot as plt

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
teacher_latent_dim = 2
student_latent_dim = 2
alpha = 0.5  # 平衡重建損失和蒸餾損失的權重
temperature = 2.0  # 溫度參數

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 載入 MNIST 資料集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 載入預訓練的教師模型
teacher_model = TeacherVAE(latent_dim=teacher_latent_dim).to(device)
teacher_model.load_state_dict(torch.load('teacher_vae_model.pth', map_location=device))
teacher_model.eval()

# 初始化學生模型
student_model = StudentVAE(latent_dim=student_latent_dim).to(device)
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

# 定義損失函數
def loss_function(recon_x, x, mu, log_var, teacher_recon, teacher_mu, teacher_log_var, 
                  alpha=0.5, beta=1.0, gamma=1.0, temperature=1.0):
    # 原始重建損失
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='mean')  # 避免 batch size 影響

    # 原始 KL 散度損失
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())  

    # 教師模型的重建損失（蒸餾）
    distill_recon_loss = F.mse_loss(recon_x, teacher_recon, reduction='mean')  # 使用 mean 避免 batch size 影響

    # KL 散度蒸餾損失
    teacher_std = torch.exp(0.5 * teacher_log_var)
    student_std = torch.exp(0.5 * log_var)

    kl_distill = 0.5 * torch.mean(
        teacher_log_var - log_var + 
        ((student_std.pow(2) + (mu - teacher_mu).pow(2)) / teacher_std.pow(2) - 1)
    ) * (1 / temperature**2)  # 溫度調整 KL loss

    # 總損失：VAE 原始損失 + 蒸餾損失
    distill_loss = beta * distill_recon_loss + gamma * kl_distill
    return BCE + KLD + alpha * distill_loss

# 訓練函數
def train(epoch):
    student_model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        
        # 獲取教師模型的輸出
        with torch.no_grad():
            teacher_recon, teacher_mu, teacher_log_var = teacher_model(data)
        
        # 訓練學生模型
        recon_batch, mu, log_var = student_model(data)
        loss = loss_function(recon_batch, data, mu, log_var,
                           teacher_recon, teacher_mu, teacher_log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)

# 測試函數
def test():
    student_model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            # 獲取教師模型的輸出
            teacher_recon, teacher_mu, teacher_log_var = teacher_model(data)
            # 獲取學生模型的輸出
            recon_batch, mu, log_var = student_model(data)
            # 計算損失
            test_loss += loss_function(recon_batch, data, mu, log_var,
                                     teacher_recon, teacher_mu, teacher_log_var).item()
    
    test_loss /= len(test_loader.dataset)
    return test_loss

# 訓練模型
train_losses = []
test_losses = []

print("開始訓練學生VAE模型...")
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss = test()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    print(f'Epoch: {epoch}')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}\n')
    
    # 每10個epoch儲存一些重建的圖像比較
    if epoch % 10 == 0:
        with torch.no_grad():
            # 從測試集取8張圖片
            data = next(iter(test_loader))[0][:8].to(device)
            
            # 獲取教師和學生模型的重建結果
            teacher_recon, _, _ = teacher_model(data)
            student_recon, _, _ = student_model(data)
            
            # 繪製原始圖片、教師重建和學生重建的比較
            plt.figure(figsize=(16, 6))
            for i in range(8):
                # 原始圖片
                plt.subplot(3, 8, i + 1)
                plt.imshow(data[i][0].cpu().numpy(), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Original')
                
                # 教師重建
                plt.subplot(3, 8, i + 9)
                plt.imshow(teacher_recon[i].view(28, 28).cpu().numpy(), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Teacher')
                
                # 學生重建
                plt.subplot(3, 8, i + 17)
                plt.imshow(student_recon[i].view(28, 28).cpu().numpy(), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Student')
            
            plt.tight_layout()
            plt.savefig(f'figures/student_vae_reconstruction_epoch_{epoch}.png')
            plt.close()

# 儲存模型
torch.save(student_model.state_dict(), 'student_vae_model.pth')

# 繪製訓練結果
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Student VAE Training Progress')
plt.tight_layout()
plt.savefig('figures/student_vae_training_loss.png')
plt.close()

# 生成一些隨機樣本並比較
with torch.no_grad():
    # 從標準正態分佈採樣
    teacher_sample = torch.randn(32, teacher_latent_dim).to(device)
    student_sample = torch.randn(32, student_latent_dim).to(device)
    
    # 解碼樣本
    teacher_output = teacher_model.decode(teacher_sample).cpu()
    student_output = student_model.decode(student_sample).cpu()
    
    # 繪製生成的圖片比較
    plt.figure(figsize=(16, 8))
    
    # 教師生成的圖片
    for i in range(32):
        plt.subplot(4, 16, i + 1)
        plt.imshow(teacher_output[i].view(28, 28).numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Teacher')
    
    # 學生生成的圖片
    for i in range(32):
        plt.subplot(4, 16, i + 33)
        plt.imshow(student_output[i].view(28, 28).numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Student')
    
    plt.tight_layout()
    plt.savefig('figures/vae_samples_comparison.png')
    plt.close() 