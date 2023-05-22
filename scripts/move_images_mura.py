"""
Script para organizar las imágenes de entrenamiento y validación del 
conjunto de datos MURA en una estructura de directorios deseada. 
Mueve las imágenes de la ubicación original a las nuevas ubicaciones 
según se especifica en los archivos CSV de rutas de imágenes.

Nota: Asegúrate de que las rutas de los archivos CSV y las rutas de destino 
estén configuradas correctamente antes de ejecutar el script.
"""
import os
import shutil

train_origin_path = os.path.join("data", "MURA-v1.1", "train_image_paths.csv")
valid_origin_path = os.path.join("data", "MURA-v1.1", "valid_image_paths.csv")

train_dest_path = os.path.join("data", "DL_images", "train")
valid_dest_path = os.path.join("data", "DL_images", "valid")

with open(train_origin_path) as f:
    img_num = 1
    lines = f.readlines()
    for line in lines:
        origin = f'data/{line.strip()}'
        print(origin)
        if os.path.exists(origin):
            shutil.move(origin, f'{train_dest_path}/image_{img_num}.png')
            img_num += 1
f.close()

with open(valid_origin_path) as g:
    img_num = 1
    lines = g.readlines()
    for line in lines:
        origin = f'data/{line.strip()}'
        if os.path.exists(origin):
            print(origin)
            shutil.move(origin, f'{valid_dest_path}/image_{img_num}.png')
            img_num += 1
g.close()