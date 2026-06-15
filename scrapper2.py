import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

sections = [
    "https://fungiatlas.com/poisonous-mushrooms/",
    "https://fungiatlas.com/edible-mushrooms/",
    "https://fungiatlas.com/inedible-mushrooms/"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_folder_name(url):
    return urlparse(url).path.strip("/").split("/")[-1]

def get_species_links(section_url):
    print(f"\n[Sección] Buscando especies en: {section_url}")
    links = set()
    try:
        response = requests.get(section_url, headers=headers)
        if response.status_code != 200: return links
        
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("article") or soup.find("main") or soup
        
        for a_tag in main_content.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(section_url, href)
            # Filtro para especies
            if "fungiatlas.com" in full_url and full_url != section_url and not full_url.endswith("/page/") and len(urlparse(full_url).path.strip("/").split("/")) == 1:
                links.add(full_url)
    except Exception as e:
        print(f"Error: {e}")
    return links

def scrape_species_gallery(species_url, target_folder):
    """Descarga imágenes directamente en la carpeta de la sección."""
    species_name = get_folder_name(species_url)
    
    print(f"   [Especie] Procesando: {species_name}")
    
    try:
        response = requests.get(species_url, headers=headers)
        if response.status_code != 200: return
            
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("article") or soup.find("main") or soup
        images = main_content.find_all("img")
        
        count = 1
        for img in images:
            img_url = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not img_url: continue
                
            full_img_url = urljoin(species_url, img_url)
            
            # Filtros básicos de limpieza
            if any(x in full_img_url.lower() for x in ["logo", "avatar", "gravatar"]) or full_img_url.endswith(".svg"):
                continue
            
            # Descarga y guardado
            try:
                img_data = requests.get(full_img_url, headers=headers).content
                ext = os.path.splitext(urlparse(full_img_url).path)[1]
                if not ext or len(ext) > 5: ext = ".jpg"
                
                # Nombre de archivo compuesto para evitar sobrescrituras
                filename = f"{species_name}_{count}{ext}"
                filepath = os.path.join(target_folder, filename)
                
                with open(filepath, "wb") as handler:
                    handler.write(img_data)
                count += 1
            except:
                continue
                
    except Exception as e:
        print(f"   Error: {e}")

# --- EJECUCIÓN ---
for section in sections:
    section_folder = get_folder_name(section)
    if not os.path.exists(section_folder):
        os.makedirs(section_folder)
    
    species_links = get_species_links(section)
    for link in species_links:
        scrape_species_gallery(link, section_folder)
        time.sleep(0.5)

print("\n¡Proceso finalizado! Estructura plana creada con éxito.")