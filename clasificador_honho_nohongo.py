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

# Carpetas raíz creadas por el scraper
sections = ["poisonous-mushrooms", "edible-mushrooms", "inedible-mushrooms"]

for section in sections:
    if not os.path.exists(section): continue
    
    print(f"\n--- Clasificando carpeta: {section} ---")
    
    # Crear subcarpetas
    ok_dir = os.path.join(section, "ok")
    ruido_dir = os.path.join(section, "ruido")
    os.makedirs(ok_dir, exist_ok=True)
    os.makedirs(ruido_dir, exist_ok=True)
    
    # Procesar archivos
    for filename in os.listdir(section):
        file_path = os.path.join(section, filename)
        
        # Solo procesar archivos (ignoramos carpetas)
        if os.path.isfile(file_path):
            if is_mushroom(file_path):
                shutil.move(file_path, os.path.join(ok_dir, filename))
                print(f"[OK] {filename}")
            else:
                shutil.move(file_path, os.path.join(ruido_dir, filename))
                print(f"[RUIDO] {filename}")

print("\nPronto!")