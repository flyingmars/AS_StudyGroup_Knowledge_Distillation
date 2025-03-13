import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import TeacherNet, StudentNet

# 設定設備
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# 指定使用第1號GPU
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print(f"使用GPU 1: {torch.cuda.get_device_name(1)}")
else:
    device = torch.device("cpu") 
    print("無法使用GPU,使用CPU替代")


# 載入資料
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 載入模型
teacher_model = TeacherNet().to(device)
student_model = StudentNet().to(device)

teacher_model.load_state_dict(torch.load('teacher_model.pth', map_location=device))
student_model.load_state_dict(torch.load('student_model.pth', map_location=device))

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

def get_predictions(model, loader):
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            if isinstance(model, TeacherNet):
                x = data.view(-1, 784)
                x = F.relu(model.fc1(x))
                x = model.dropout(x)
                x = F.relu(model.fc2(x))
                features = model.dropout(x)
            else:
                x = data.view(-1, 784)
                features = F.relu(model.fc1(x))
            
            all_features.extend(features.cpu().numpy())
    
    return np.array(all_preds).reshape(-1), np.array(all_labels), np.array(all_features)

# 獲取預測結果
teacher_preds, true_labels, teacher_features = get_predictions(teacher_model, test_loader)
student_preds, _, student_features = get_predictions(student_model, test_loader)

# 計算混淆矩陣
teacher_cm = confusion_matrix(true_labels, teacher_preds)
student_cm = confusion_matrix(true_labels, student_preds)

def plot_confusion_matrix(cm, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

# 繪製混淆矩陣
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plot_confusion_matrix(teacher_cm, 'Teacher Model Confusion Matrix')

plt.subplot(1, 2, 2)
plot_confusion_matrix(student_cm, 'Student Model Confusion Matrix')

plt.tight_layout()
plt.savefig('figures/confusion_matrices.png')
plt.close()

def plot_tsne(features, labels, preds, title):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'{title} (True Labels)')
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=preds, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'{title} (Predicted Labels)')
    
    plt.tight_layout()
    return features_2d

# 繪製t-SNE可視化
print("Performing t-SNE dimensionality reduction, this may take a while...")
teacher_features_2d = plot_tsne(teacher_features, true_labels, teacher_preds, 'Teacher Model Feature Distribution')
plt.savefig('figures/teacher_tsne.png')
plt.close()

student_features_2d = plot_tsne(student_features, true_labels, student_preds, 'Student Model Feature Distribution')
plt.savefig('figures/student_tsne.png')
plt.close()

# 計算並顯示模型性能指標
teacher_accuracy = (teacher_preds == true_labels).mean() * 100
student_accuracy = (student_preds == true_labels).mean() * 100

print("\nModel Performance Comparison:")
print(f"Teacher Model Accuracy: {teacher_accuracy:.2f}%")
print(f"Student Model Accuracy: {student_accuracy:.2f}%")

# 分析預測差異
disagreements = (teacher_preds != student_preds).sum()
print(f"\nNumber of prediction disagreements between teacher and student: {disagreements}")
print(f"Disagreement ratio: {disagreements/len(true_labels)*100:.2f}%")

# 儲存預測結果比較
plt.figure(figsize=(15, 5))
sample_size = 50
indices = np.random.choice(len(true_labels), sample_size, replace=False)

plt.subplot(1, 1, 1)
width = 0.35
x = np.arange(sample_size)
plt.bar(x - width/2, teacher_preds[indices], width, label='Teacher Predictions', alpha=0.6)
plt.bar(x + width/2, student_preds[indices], width, label='Student Predictions', alpha=0.6)
plt.plot(x, true_labels[indices], 'r*', label='True Labels')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Teacher vs Student Model Predictions (Random Samples)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/prediction_comparison.png')
plt.close() 