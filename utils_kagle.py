import kagglehub
import shutil
import os

# Descargar dataset
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

print("Path to dataset files:", path)

# Definir carpeta destino
dest_folder = "Dataset"

# Copiar dataset descargado a carpeta Dataset/
if not os.path.exists(dest_folder):
    shutil.copytree(path, dest_folder)
    print(f"Dataset copiado en {dest_folder}")
else:
    print("Dataset ya existe en carpeta local.")
