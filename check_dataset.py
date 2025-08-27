import os
import kagglehub

path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("Dataset descargado en:", path)

# Listar lo que hay dentro
print("\nContenido de la carpeta raíz:")
print(os.listdir(path))

for item in os.listdir(path):
    item_path = os.path.join(path, item)
    if os.path.isdir(item_path):
        print(f"\nSubcarpeta encontrada: {item}")
        print(os.listdir(item_path)[:10])  # muestra solo 10 elementos
