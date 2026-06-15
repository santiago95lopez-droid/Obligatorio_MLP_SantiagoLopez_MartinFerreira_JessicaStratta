import os
import random
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
model = MobileNetV2(weights='imagenet')

# Imagenes de ruido
folders_to_audit = ["poisonous-mushrooms", "edible-mushrooms", "inedible-mushrooms"]
all_ruido_images = []

for section in folders_to_audit:
    ruido_path = os.path.join(section, "ruido")
    if os.path.exists(ruido_path):
        for filename in os.listdir(ruido_path):

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_ruido_images.append(os.path.join(ruido_path, filename))

print(f"Total {len(all_ruido_images)} imágenes en las carpetas 'ruido'.")

#aca es donde selecciono 10 casos al azar para auditar manualmente. Si hay menos de 10, se auditan todas. Si no hay ninguna, se informa y se salta la auditoría.
num_samples = min(10, len(all_ruido_images))

if num_samples == 0:
    print("0 imagenes")
else:
    random.seed(42)  
    selected_images = random.sample(all_ruido_images, num_samples)
    
    print(f"\n=== AUDITORÍA DE {num_samples} CASOS AL AZAR EN 'RUIDO' ===")
    
    for idx, img_path in enumerate(selected_images, 1):
        print(f"\n[{idx}/10] Archivo: {img_path}")
        try:
            # Preprocesamiento clásico
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            
            preds = model.predict(x, verbose=0)
            decoded = decode_predictions(preds, top=3)[0]
            
            
            for rank, (imagenet_id, label, prob) in enumerate(decoded, 1):
                print(f"    Top {rank}: {label} ({prob * 100:.2f}%)")
                
        except Exception as e:
            print(f"    No se pudo analizar esta imagen: {e}")