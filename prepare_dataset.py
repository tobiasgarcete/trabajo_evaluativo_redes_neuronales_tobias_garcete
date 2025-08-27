import os
import shutil
import kagglehub
import random

if os.path.exists("Dataset"):
    shutil.rmtree("Dataset")
# 0. Definir máximo de imágenes por clase
MAX_IMAGES_PER_CLASS = 30  # máximo 30 imágenes por clase

# 1. Descargar dataset
print("Descargando dataset desde Kaggle...")
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("Dataset descargado en:", path)

# 2. Ajustar la ruta real (usamos las imágenes a color)
source_folder = os.path.join(path, "plantvillage dataset", "color")

if not os.path.exists(source_folder):
    raise FileNotFoundError(f"No se encontró la carpeta {source_folder}. Revisa la estructura del dataset.")

# 3. Definir carpetas destino
DEST_DIR = "Dataset"
TRAIN_DIR = os.path.join(DEST_DIR, "train")
TEST_DIR = os.path.join(DEST_DIR, "test")

# Crear estructura de carpetas
for folder in [TRAIN_DIR, TEST_DIR]:
    for label in ["sano", "plaga"]:
        os.makedirs(os.path.join(folder, label), exist_ok=True)

# 4. Recorrer carpetas del dataset original
all_classes = os.listdir(source_folder)
print("Clases encontradas (ejemplo):", all_classes[:5], "...")

# 5. Clasificar imágenes en sano vs plaga
for cls in all_classes:
    cls_path = os.path.join(source_folder, cls)
    images = os.listdir(cls_path)

    # Barajar imágenes para dividir train/test
    random.shuffle(images)
    split_idx = int(0.8 * len(images))  # 80% train, 20% test
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    # ⚡ Limitar la cantidad de imágenes por clase
    train_imgs = train_imgs[:MAX_IMAGES_PER_CLASS]
    test_imgs = test_imgs[:int(MAX_IMAGES_PER_CLASS * 0.2)]  # mantener proporción 80/20

    # Determinar etiqueta (sano/plaga)
    label = "sano" if "healthy" in cls.lower() else "plaga"

    # Copiar imágenes
    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(TRAIN_DIR, label, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(TEST_DIR, label, img))

print("✅ Dataset reorganizado en carpetas 'sano' y 'plaga'.")
print("Ruta final:", DEST_DIR)
