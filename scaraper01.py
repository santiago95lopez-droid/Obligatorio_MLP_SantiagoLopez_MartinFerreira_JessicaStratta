import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Defino las 3 urls que quiero scrappear. Cada una corresponde a una sección diferente de la web.
urls_to_scrape = [
    "https://fungiatlas.com/poisonous-mushrooms/",
    "https://fungiatlas.com/edible-mushrooms/",
    "https://fungiatlas.com/inedible-mushrooms/"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_section(url):
    print(f"\n--- Iniciando scraping de: {url} ---")
    
    # Obtener el nombre de la carpeta a partir del final de la url
    parsed_url = urlparse(url)
    folder_name = parsed_url.path.strip("/").split("/")[-1]
    
    # Si por alguna razón queda vacío, le asigno un nombre generico
    if not folder_name:
        folder_name = "imagenes_hongos"
        
    # Creamos la carpeta
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Carpeta creada : '{folder_name}'")
    else:
        print(f"Usando carpeta : '{folder_name}'")

    print("Conectando")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error al acceder a {url}. Estado: {response.status_code}")
            return
            
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("article") or soup.find("main") or soup
        images = main_content.find_all("img")
        
        print(f"Se encontraron {len(images)} imágenes")
        
        count = 1
        for img in images:
            img_url = img.get("src") or img.get("data-src")
            if not img_url:
                continue
                
            img_url = urljoin(url, img_url)
            
            if "logo" in img_url.lower() or "avatar" in img_url.lower() or img_url.endswith(".svg"):
                continue
                
            try:
                img_data = requests.get(img_url, headers=headers).content
                filename = f"hongo_{count}.jpg"
                filepath = os.path.join(folder_name, filename)
                
                with open(filepath, "wb") as handler:
                    handler.write(img_data)
                    
                count += 1
            except Exception as e:
                print(f"Error en {img_url}: {e}")
                
        print(f"Sección completada {count - 1} imágenes guardadas en '{folder_name}'")
        
    except Exception as e:
        print(f"Error general procesando la URL {url}: {e}")

# Se ejecuta el scraping para cada una de las urls definidas al inicio
for target_url in urls_to_scrape:
    scrape_section(target_url)

print("\n Pronto!")