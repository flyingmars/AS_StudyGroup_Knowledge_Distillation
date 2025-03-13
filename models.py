import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import os
import graphviz
import torchviz
from torchsummary import summary
import sys

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # 返回原始 logits
    
    def predict(self, x):
        # 用於預測時獲取概率分布
        return F.log_softmax(self.forward(x), dim=1)

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 返回原始 logits
    
    def predict(self, x):
        # 用於預測時獲取概率分布
        return F.log_softmax(self.forward(x), dim=1)

class TeacherVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(TeacherVAE, self).__init__()
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_var = nn.Linear(400, latent_dim)
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class StudentVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(StudentVAE, self).__init__()
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(784, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(50, latent_dim)
        self.fc_var = nn.Linear(50, latent_dim)
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 784),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def create_network_graph(model, name):
    dot = graphviz.Digraph(name, format='png')
    dot.attr(rankdir='LR')  # 從左到右的布局
    dot.attr('node', shape='box', style='rounded')
    
    # 為模型添加節點和邊
    if isinstance(model, TeacherNet):
        dot.node('input', 'Input\n(784)')
        dot.node('fc1', 'Linear\n(784→1200)')
        dot.node('dropout1', 'Dropout\n(0.3)')
        dot.node('fc2', 'Linear\n(1200→1200)')
        dot.node('dropout2', 'Dropout\n(0.3)')
        dot.node('fc3', 'Linear\n(1200→10)')
        dot.node('output', 'Output\n(10)')
        
        dot.edge('input', 'fc1')
        dot.edge('fc1', 'dropout1')
        dot.edge('dropout1', 'fc2')
        dot.edge('fc2', 'dropout2')
        dot.edge('dropout2', 'fc3')
        dot.edge('fc3', 'output')
        
    elif isinstance(model, StudentNet):
        dot.node('input', 'Input\n(784)')
        dot.node('fc1', 'Linear\n(784→10)')
        dot.node('fc2', 'Linear\n(10→10)')
        dot.node('output', 'Output\n(10)')
        
        dot.edge('input', 'fc1')
        dot.edge('fc1', 'fc2')
        dot.edge('fc2', 'output')
        
    elif isinstance(model, (TeacherVAE, StudentVAE)):
        hidden_dim = '400' if isinstance(model, TeacherVAE) else '50'
        
        # Encoder
        dot.node('input', 'Input\n(784)')
        dot.node('enc_fc1', f'Linear\n(784→{hidden_dim})')
        dot.node('enc_fc2', f'Linear\n({hidden_dim}→{hidden_dim})')
        dot.node('fc_mu', f'Linear\n({hidden_dim}→2)')
        dot.node('fc_var', f'Linear\n({hidden_dim}→2)')
        dot.node('reparameterize', 'Reparameterize')
        
        # Decoder
        dot.node('dec_fc1', f'Linear\n(2→{hidden_dim})')
        dot.node('dec_fc2', f'Linear\n({hidden_dim}→{hidden_dim})')
        dot.node('dec_fc3', f'Linear\n({hidden_dim}→784)')
        dot.node('output', 'Output\n(784)')
        
        # Encoder edges
        dot.edge('input', 'enc_fc1')
        dot.edge('enc_fc1', 'enc_fc2')
        dot.edge('enc_fc2', 'fc_mu')
        dot.edge('enc_fc2', 'fc_var')
        dot.edge('fc_mu', 'reparameterize')
        dot.edge('fc_var', 'reparameterize')
        
        # Decoder edges
        dot.edge('reparameterize', 'dec_fc1')
        dot.edge('dec_fc1', 'dec_fc2')
        dot.edge('dec_fc2', 'dec_fc3')
        dot.edge('dec_fc3', 'output')
    
    dot.attr(dpi='300')
    return dot

if __name__ == "__main__":
    # 創建模型實例
    teacher_net = TeacherNet()
    student_net = StudentNet()
    teacher_vae = TeacherVAE()
    student_vae = StudentVAE()
    
    # 確保 figures 目錄存在
    os.makedirs('figures', exist_ok=True)
    
    # 生成並保存模型結構圖
    create_network_graph(teacher_net, "TeacherNet").render("figures/teacher_net_architecture")
    create_network_graph(student_net, "StudentNet").render("figures/student_net_architecture")
    create_network_graph(teacher_vae, "TeacherVAE").render("figures/teacher_vae_architecture")
    create_network_graph(student_vae, "StudentVAE").render("figures/student_vae_architecture")
    
    print("模型結構圖已生成，請查看 figures 目錄下的 .png 文件") 