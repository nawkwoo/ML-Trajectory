"""
간단한 LSTM 분류 모델
Author: 윤형
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
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
# 모델
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

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
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
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
            
            outputs = model(X_batch)
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
print("LSTM 분류 모델 학습")
print("="*60)

# 데이터 로딩
print("\n[1] 데이터 로딩")
data_dir = Path("augmented_data")

if not data_dir.exists():
    print("❌ augmented_data/ 없음!")
    print("먼저 실행: python src/my_preprocess.py")
    exit(1)

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 모델
print("\n[2] 모델 생성")
model = LSTMClassifier(input_dim=3, hidden_dim=64, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 학습
print("\n[3] 학습 시작")
best_acc = 0
history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(50):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_acc, _, _ = evaluate(model, test_loader)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    
    if test_acc > best_acc:
        best_acc = test_acc
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

# 최종 평가
print("\n[4] 최종 평가")
test_acc, y_pred, y_true = evaluate(model, test_loader)
print(f"  Test Accuracy: {test_acc:.4f}")

# Classification Report
class_names = ['circle', 'diagonal_left', 'diagonal_right', 'horizontal', 'vertical']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 학습 곡선
print("\n[5] 학습 곡선 저장")
plt.figure(figsize=(10, 6))
epochs = range(1, len(history['train_loss']) + 1)
plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
plt.plot(epochs, history['test_acc'], 'r-', label='Test Acc', linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Training History', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/simple_model_history.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/figures/simple_model_history.png")
plt.close()

# 모델 저장
print("\n[6] 모델 저장")
Path("models").mkdir(exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'X_mean': X_mean,
    'X_std': X_std,
    'class_names': class_names,
    'test_acc': test_acc
}, 'models/simple_model.pt')
print("✓ 저장: models/simple_model.pt")

print("\n" + "="*60)
print("✓ 완료!")
print(f"  Best Test Accuracy: {best_acc:.4f}")
print("="*60)