import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import TeacherVAE, StudentVAE
from sklearn.manifold import TSNE

# 檢查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print(f"使用GPU 1: {torch.cuda.get_device_name(1)}")
else:
    device = torch.device("cpu") 
    print("無法使用GPU,使用CPU替代")

# 設定參數
latent_dim = 2
n_points = 10  # 減少採樣點數以適應繪圖
range_min, range_max = -3, 3

# 載入模型
teacher_model = TeacherVAE(latent_dim=latent_dim).to(device)
student_model = StudentVAE(latent_dim=latent_dim).to(device)

teacher_model.load_state_dict(torch.load('teacher_vae_model.pth', map_location=device))
student_model.load_state_dict(torch.load('student_vae_model.pth', map_location=device))

# 計算並顯示模型參數數量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

teacher_params = count_parameters(teacher_model)
student_params = count_parameters(student_model)
params_reduction = (1 - student_params / teacher_params) * 100

print("\n模型參數統計:")
print(f"教師模型參數數量: {teacher_params:,}")
print(f"學生模型參數數量: {student_params:,}")
print(f"參數減少比例: {params_reduction:.2f}%")

teacher_model.eval()
student_model.eval()

# 創建潛在空間的網格點
x = np.linspace(range_min, range_max, n_points)
y = np.linspace(range_min, range_max, n_points)
xx, yy = np.meshgrid(x, y)

# 將網格點轉換為張量
grid = torch.FloatTensor(np.column_stack([xx.ravel(), yy.ravel()])).to(device)

# 生成和比較重建結果
with torch.no_grad():
    # 教師模型生成
    teacher_output = teacher_model.decode(grid).cpu()
    
    # 學生模型生成
    student_output = student_model.decode(grid).cpu()

# 分別繪製教師和學生模型的結果
def plot_model_outputs(output, title, filename):
    plt.figure(figsize=(15, 15))
    for i in range(n_points * n_points):
        plt.subplot(n_points, n_points, i + 1)
        plt.imshow(output[i].view(28, 28).numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 繪製教師模型的結果
plot_model_outputs(teacher_output, 'Teacher Model Latent Space Generation', 
                  'figures/teacher_vae_latent_space.png')

# 繪製學生模型的結果
plot_model_outputs(student_output, 'Student Model Latent Space Generation', 
                  'figures/student_vae_latent_space.png')

# 載入測試數據
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 收集編碼結果
def collect_encodings(model, loader):
    encodings = []
    labels = []
    reconstructions = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            recon, _, _ = model(data)
            encodings.append(mu.cpu().numpy())
            labels.extend(target.numpy())
            reconstructions.append(recon.cpu().numpy())
    
    return np.concatenate(encodings), np.array(labels), np.concatenate(reconstructions)

# 獲取編碼結果
teacher_encodings, labels, teacher_reconstructions = collect_encodings(teacher_model, test_loader)
student_encodings, _, student_reconstructions = collect_encodings(student_model, test_loader)

# 繪製潛在空間分佈
plt.figure(figsize=(20, 10))

# 教師模型的潛在空間
plt.subplot(1, 2, 1)
scatter = plt.scatter(teacher_encodings[:, 0], teacher_encodings[:, 1], 
                     c=labels, cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.title('Teacher Model Latent Space')
plt.xlabel('First Latent Dimension')
plt.ylabel('Second Latent Dimension')

# 學生模型的潛在空間
plt.subplot(1, 2, 2)
scatter = plt.scatter(student_encodings[:, 0], student_encodings[:, 1], 
                     c=labels, cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.title('Student Model Latent Space')
plt.xlabel('First Latent Dimension')
plt.ylabel('Second Latent Dimension')

plt.tight_layout()
plt.savefig('figures/vae_encodings_comparison.png')
plt.close()

# 計算重建誤差
teacher_mse = np.mean((teacher_reconstructions - test_loader.dataset.data.numpy().reshape(-1, 784) / 255.0) ** 2)
student_mse = np.mean((student_reconstructions - test_loader.dataset.data.numpy().reshape(-1, 784) / 255.0) ** 2)

print(f"教師模型重建MSE: {teacher_mse:.4f}")
print(f"學生模型重建MSE: {student_mse:.4f}")

# 創建潛在空間遍歷的動畫幀
def create_latent_traversal(z1_range, z2_value):
    z1 = torch.linspace(range_min, range_max, n_points).to(device)
    z2 = torch.ones_like(z1).to(device) * z2_value
    z = torch.stack([z1, z2], dim=1)
    
    with torch.no_grad():
        teacher_images = teacher_model.decode(z).cpu()
        student_images = student_model.decode(z).cpu()
    
    return teacher_images, student_images

# 在不同的 z2 值下遍歷 z1
z2_values = [-2, -1, 0, 1, 2]

# 為每個 z2 值創建單獨的圖
for i, z2 in enumerate(z2_values):
    teacher_images, student_images = create_latent_traversal([-3, 3], z2)
    
    plt.figure(figsize=(15, 3))
    plt.suptitle(f'Latent Space Traversal (z2 = {z2})')
    
    # 繪製教師模型的結果
    for j in range(n_points):
        plt.subplot(2, n_points, j + 1)
        plt.imshow(teacher_images[j].view(28, 28), cmap='gray')
        plt.axis('off')
        if j == 0:
            plt.title('Teacher')
    
    # 繪製學生模型的結果
    for j in range(n_points):
        plt.subplot(2, n_points, j + n_points + 1)
        plt.imshow(student_images[j].view(28, 28), cmap='gray')
        plt.axis('off')
        if j == 0:
            plt.title('Student')
    
    plt.tight_layout()
    plt.savefig(f'figures/vae_latent_traversal_z2_{z2}.png')
    plt.close()

# 計算並比較潛在空間的統計特性
teacher_mean = np.mean(teacher_encodings, axis=0)
teacher_std = np.std(teacher_encodings, axis=0)
student_mean = np.mean(student_encodings, axis=0)
student_std = np.std(student_encodings, axis=0)

print("\n潛在空間統計比較:")
print(f"教師模型 - 平均值: {teacher_mean}, 標準差: {teacher_std}")
print(f"學生模型 - 平均值: {student_mean}, 標準差: {student_std}")

# 為每個數字類別計算平均潛在向量
def compute_class_centroids(encodings, labels):
    centroids = {}
    for i in range(10):
        mask = labels == i
        centroids[i] = np.mean(encodings[mask], axis=0)
    return centroids

teacher_centroids = compute_class_centroids(teacher_encodings, labels)
student_centroids = compute_class_centroids(student_encodings, labels)

# 繪製類別中心點
plt.figure(figsize=(20, 10))

# 教師模型的類別中心
plt.subplot(1, 2, 1)
for digit in range(10):
    centroid = teacher_centroids[digit]
    plt.scatter(centroid[0], centroid[1], s=100, label=str(digit))
plt.title('Teacher Model Class Centroids')
plt.legend()

# 學生模型的類別中心
plt.subplot(1, 2, 2)
for digit in range(10):
    centroid = student_centroids[digit]
    plt.scatter(centroid[0], centroid[1], s=100, label=str(digit))
plt.title('Student Model Class Centroids')
plt.legend()

plt.tight_layout()
plt.savefig('figures/vae_class_centroids.png')
plt.close() 