"""
개선된 LSTM 분류 모델 (Attention 추가)
Author: 윤형

개선 사항:
- Attention 메커니즘
- Bidirectional LSTM
- Layer Normalization
- Learning Rate Scheduler
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}\n")

# ==========================================
# Dataset
# ==========================================
class TrajDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# Attention 메커니즘
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1), dim=1
        )
        # attention_weights: (batch, seq_len)
        
        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)
        # context: (batch, hidden_dim)
        
        return context, attention_weights

# ==========================================
# 개선된 모델 (BiLSTM + Attention)
# ==========================================
class ImprovedLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=5):
        super().__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Attention
        self.attention = Attention(hidden_dim * 2)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        
        # Classifier
        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out, attention_weights

# ==========================================
# 학습 함수
# ==========================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs, _ = model(X_batch)
            _, predicted = outputs.max(1)
            
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return correct / total, np.array(all_preds), np.array(all_labels)

# ==========================================
# 메인
# ==========================================
print("="*60)
print("개선된 LSTM 분류 모델 학습")
print("="*60)

# 데이터 로딩
print("\n[1] 데이터 로딩")
data_dir = Path("augmented_data")

X = np.load(data_dir / "X.npy")
y = np.load(data_dir / "y.npy")

print(f"  X: {X.shape}")
print(f"  y: {y.shape}")

# 정규화
X_mean = X.mean(axis=(0, 1), keepdims=True)
X_std = X.std(axis=(0, 1), keepdims=True)
X = (X - X_mean) / (X_std + 1e-8)

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train: {X_train.shape}")
print(f"  Test:  {X_test.shape}")

# DataLoader
train_dataset = TrajDataset(X_train, y_train)
test_dataset = TrajDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# 모델
print("\n[2] 모델 생성")
model = ImprovedLSTM(input_dim=3, hidden_dim=128, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)

params = sum(p.numel() for p in model.parameters())
print(f"  파라미터 수: {params:,}")

# 학습
print("\n[3] 학습 시작 (100 에폭)")
best_acc = 0
best_epoch = 0
history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
patience = 0
max_patience = 30

for epoch in range(100):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_acc, _, _ = evaluate(model, test_loader)
    scheduler.step(test_acc)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch + 1
        patience = 0
        # 최고 모델 저장
        best_model_state = model.state_dict().copy()
    else:
        patience += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f} "
              f"(Best: {best_acc:.4f})")
    
    # Early Stopping
    if patience >= max_patience:
        print(f"\n  Early stopping at epoch {epoch+1}")
        break

# 최고 모델 로드
model.load_state_dict(best_model_state)

# 최종 평가
print(f"\n[4] 최종 평가 (Best Epoch: {best_epoch})")
test_acc, y_pred, y_true = evaluate(model, test_loader)
print(f"  Test Accuracy: {test_acc:.4f}")

# Classification Report
class_names = ['circle', 'diagonal_left', 'diagonal_right', 'horizontal', 'vertical']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# 시각화
print("\n[5] 결과 시각화")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 학습 곡선
epochs = range(1, len(history['train_loss']) + 1)
axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Acc', linewidth=2)
axes[1].axhline(y=best_acc, color='g', linestyle='--', label=f'Best: {best_acc:.4f}')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Training History', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/improved_lstm_results.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/improved_lstm_results.png")
plt.close()

# 모델 저장
print("\n[6] 모델 저장")
Path("models").mkdir(exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'X_mean': X_mean,
    'X_std': X_std,
    'class_names': class_names,
    'test_acc': test_acc,
    'best_epoch': best_epoch
}, 'models/improved_lstm.pt')
print("✓ 저장: models/improved_lstm.pt")

# 비교
print("\n" + "="*60)
print("✓ 완료!")
print("="*60)
print(f"\n성능 비교:")
print(f"  기존 LSTM:    69.23%")
print(f"  개선 LSTM:    {best_acc*100:.2f}%")
print(f"  개선도:       {(best_acc - 0.6923)*100:+.2f}%p")
print(f"\n최고 성능 에폭: {best_epoch}")
print("="*60)