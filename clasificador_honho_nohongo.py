import os
import shutil
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

#MobileNetV2 pre-entrenado con pesos de ImageNet
model = MobileNetV2(weights='imagenet')

# keywords de hongos en el mobilenet
MUSHROOM_KEYWORDS = ['mushroom', 'fungus', 'agaric', 'boletus', 'puffball', 'chanterelle', 'morel', 'truffle', 'coral fungus', 'earthstar', 'stinkhorn', 'toadstool'
                     'stinkhorn','bolete']

def is_mushroom(img_path):
    try:
        # Preparamos la imagen
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        
        preds = model.predict(x, verbose=0)
        # Decodificamos las 3 predicciones principales
        decoded = decode_predictions(preds, top=3)[0]
        
        # Si alguno de los resultados (descripción) contiene palabras clave, es un hongo
        for _, label, prob in decoded:
            if any(keyword in label.lower() for keyword in MUSHROOM_KEYWORDS):
                return True
        return False
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")
        return False

BASE_DIR = r"C:\Users\marti\Desktop\Master\H3\Machine Learning en Produccion\scrapper_fungiatlas\hongos_scrapp"
RUEDO_DIR = os.path.join(BASE_DIR, "ruido")

if not os.path.exists(BASE_DIR):
    print(f"No se encontró la carpeta de imágenes: {BASE_DIR}")
else:
    os.makedirs(RUEDO_DIR, exist_ok=True)
    print(f"\n--- Clasificando imágenes en: {BASE_DIR} ---")

    for filename in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, filename)

        if not os.path.isfile(file_path):
            continue
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        if is_mushroom(file_path):
            print(f"[OK] {filename}")
        else:
            target_path = os.path.join(RUEDO_DIR, filename)
            shutil.move(file_path, target_path)
            print(f"[RUIDO] {filename}")

print("\nPronto!")