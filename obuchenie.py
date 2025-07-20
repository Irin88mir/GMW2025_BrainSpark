import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
from tqdm import tqdm

#Обучение модели Inception

# 1. Определяем устройство в самом начале
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Функция создания модели
def create_model():
    model = models.inception_v3(weights='DEFAULT', aux_logits=True)

    # Улучшенный классификатор
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.7),
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 2)
    )

    # Вспомогательный классификатор
    num_aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_aux_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 2)
    )

    # Замораживаем слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем нужные слои
    for name, param in model.named_parameters():
        if 'fc' in name or 'AuxLogits' in name or 'Mixed_7' in name:
            param.requires_grad = True

    return model.to(device)


# 3. Аугментация данных
transform = transforms.Compose([
    transforms.Resize(342),
    transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Загрузка данных
dataset = datasets.ImageFolder("data_delay_cropped_binar/", transform=transform)
train_data, test_data = random_split(dataset, [0.7, 0.3])   

# 5. Веса классов
class_counts = torch.bincount(torch.tensor([y for _, y in train_data]))
class_weights = (1.0 / class_counts.float()).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# 6. DataLoader'ы
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 7. Инициализация модели
model = create_model()

# 8. Оптимизатор
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.01,
    weight_decay=1e-3
)

# 9. Планировщик
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=1000
)


# 10. Функция обучения
def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs):
    best_f1 = 0.0
    best_epoch = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=True)
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(X)
            loss = loss_fn(outputs[0] if isinstance(outputs, tuple) else outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())

        model.eval()
        test_loss = 0.0
        all_preds, all_targets = [], []
        test_iter = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", leave=True)
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = torch.argmax(outputs[0] if isinstance(outputs, tuple) else outputs, dim=1)
                test_loss += loss_fn(outputs[0] if isinstance(outputs, tuple) else outputs, y).item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                test_iter.set_postfix(loss=loss_fn(outputs[0] if isinstance(outputs, tuple) else outputs, y).item())

        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, F1 = {epoch_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_epoch = epoch + 1
            save_path = f'best_Inception_bi_{str(best_f1).replace('.', '_')}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with F1: {best_f1:.4f} (Epoch {best_epoch})")

        #if epoch - best_epoch >= patience:
            #print(f"\nEarly stopping: No improvement for {patience} epochs. Best F1: {best_f1:.4f}")
            #break

        scheduler.step(epoch_f1)


# 11. Запуск обучения
train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=150,
)
