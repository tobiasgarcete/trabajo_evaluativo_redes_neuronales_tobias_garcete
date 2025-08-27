# mlp_evaluation.py (o puedes poner esto directamente en mlp_model.py si lo prefieres)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# Definición de la clase MLPModel (asegúrate de que esto esté en mlp_model.py o aquí)
class MLPModel(nn.Module):
    def __init__(self, input_size=(3, 128, 128), num_classes=2):
        super(MLPModel, self).__init__()
        # Calcula el tamaño de entrada a la primera capa lineal después de aplanar
        # input_size es (channels, height, width)
        flattened_size = input_size[0] * input_size[1] * input_size[2]
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes) # num_classes es 2 (sano, plaga)

    def forward(self, x):
        # Aplanar la imagen: de (batch_size, channels, height, width) a (batch_size, channels * height * width)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 1. Transformaciones de imágenes (igual que CNN, pero el MLP lo aplanará después)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 2. Cargar dataset (igual que CNN)
# Asegúrate de que las carpetas "Dataset/train" y "Dataset/test" existan
# y contengan tus subcarpetas "sano" y "plaga" con las imágenes.
train_dataset = datasets.ImageFolder("Dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("Dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 3. Definir modelo MLP
# Asegúrate de pasar el tamaño correcto de la imagen (3 canales, 128x128)
# y el número de clases (2: sano, plaga)
model_mlp = MLPModel(input_size=(3, 128, 128), num_classes=len(train_dataset.classes))

# 4. Definir pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.001)

# 5. Entrenamiento (similar a CNN)
epochs_mlp = 10
print("\nEntrenando Modelo MLP...")
for epoch in range(epochs_mlp):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer_mlp.zero_grad()
        outputs = model_mlp(images) # El forward del MLP ya aplana la imagen
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_mlp.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs_mlp}], Loss (MLP): {running_loss:.4f}")

# 6. Evaluación (similar a CNN)
y_true_mlp, y_pred_mlp = [], []
model_mlp.eval() # Poner el modelo en modo de evaluación
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_mlp(images)
        _, predicted = torch.max(outputs, 1)
        y_true_mlp.extend(labels.numpy())
        y_pred_mlp.extend(predicted.numpy())

acc_mlp = accuracy_score(y_true_mlp, y_pred_mlp)
print(f"Accuracy en test (MLP): {acc_mlp*100:.2f}%")