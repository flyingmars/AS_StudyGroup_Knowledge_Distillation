import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import TeacherNet, StudentNet
from tqdm import tqdm
import matplotlib.pyplot as plt

# 檢查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 設定超參數
batch_size = 128
epochs = 10
learning_rate = 0.001
temperature = 4
alpha = 0.5  # 軟標籤和硬標籤的權重比例

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 載入 MNIST 資料集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 載入預訓練的教師模型
teacher_model = TeacherNet().to(device)
teacher_model.load_state_dict(torch.load('teacher_model.pth', map_location=device))
teacher_model.eval()

# 初始化學生模型
student_model = StudentNet().to(device)
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    計算 knowledge distillation 損失
    參數:
    - student_logits: 學生模型的原始輸出（未經過 softmax）
    - teacher_logits: 教師模型的原始輸出（未經過 softmax）
    - labels: 真實標籤
    - T: 溫度參數
    - alpha: 軟標籤權重
    """
    # 計算軟標籤損失
    student_soft = F.log_softmax(student_logits/T, dim=1)
    teacher_soft = F.softmax(teacher_logits/T, dim=1)
    soft_targets_loss = -torch.sum(teacher_soft * student_soft) / student_soft.size(0) * (T**2)
    
    # 計算硬標籤損失
    hard_targets_loss = F.cross_entropy(student_logits, labels)
    
    # 計算蒸餾損失
    # 前面是軟標籤，公式為 -1/N * sum(soft_prob * soft_targets) * T^2，為教師模型和學生模型的 KL 散度
    # 後面是硬標籤，公式為 -1/N * sum(log(soft_prob) * labels)，為學生模型自己與硬標籤的交叉熵
    return soft_targets_loss * alpha + hard_targets_loss * (1. - alpha)

# 訓練函數
def train(epoch):
    student_model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        
        # 取得教師模型的原始輸出（logits）
        with torch.no_grad():
            teacher_logits = teacher_model(data)
        
        # 訓練學生模型
        optimizer.zero_grad()
        student_logits = student_model(data)
        
        # 計算 distillation loss
        loss = distillation_loss(student_logits, teacher_logits, target, 
                               temperature, alpha)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 測試函數
def test():
    student_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 使用 predict 方法獲取 log_softmax 輸出
            output = student_model.predict(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_dataset)
    return test_loss, accuracy

# 訓練模型
train_losses = []
test_losses = []
test_accuracies = []

print("開始訓練學生模型...")
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss, accuracy = test()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    
    print(f'Epoch: {epoch}')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%\n')

# 儲存模型
torch.save(student_model.state_dict(), 'student_model.pth')

# 繪製訓練結果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('figures/student_training_results.png')
plt.close() 