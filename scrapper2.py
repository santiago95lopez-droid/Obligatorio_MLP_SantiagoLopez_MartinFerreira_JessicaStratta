import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

START_URL = "https://fungiatlas.com/all-mushrooms/"
OUTPUT_DIR = r"C:\Users\marti\Desktop\Master\H3\Machine Learning en Produccion\scrapper_fungiatlas\hongos_scrapp"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join("_" if ch in invalid_chars else ch for ch in name)
    sanitized = sanitized.strip().replace(" ", "_")
    return sanitized[:255] if len(sanitized) > 255 else sanitized


def get_species_name(species_url: str, soup: BeautifulSoup) -> str:
    parsed = urlparse(species_url)
    path = parsed.path.strip("/")
    if path:
        slug = path.split("/")[-1]
        if slug:
            return sanitize_filename(slug)

    heading = None
    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading and heading.get_text(strip=True):
            return sanitize_filename(heading.get_text(" ", strip=True))

    return "mushroom"


def get_species_links(start_url: str):
    print(f"[Inicio] Obteniendo enlaces de especies desde: {start_url}")
    species_links = []

    try:
        response = requests.get(start_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(start_url, href)
            if "fungiatlas.com" not in full_url:
                continue
            if full_url == start_url:
                continue
            if not full_url.endswith("/"):
                continue

            parsed = urlparse(full_url)
            path_parts = [part for part in parsed.path.split("/") if part]
            if len(path_parts) == 1:
                species_links.append(full_url)
    except Exception as e:
        print(f"Error al obtener enlaces: {e}")

    return sorted(set(species_links))


def find_image_candidates(soup: BeautifulSoup):
    carousel_containers = []
    for attr in ["carousel", "gallery", "slider", "swiper"]:
        for element in soup.find_all(True, class_=lambda value: value and attr in str(value).lower()):
            carousel_containers.append(element)
        for element in soup.find_all(True, id=lambda value: value and attr in str(value).lower()):
            carousel_containers.append(element)

    candidates = []
    if carousel_containers:
        for container in carousel_containers:
            for img in container.find_all("img"):
                candidates.append(img)
    else:
        for img in soup.find_all("img"):
            candidates.append(img)

    return candidates


def extract_mushroom_images(species_url: str):
    print(f"\n[Especie] Procesando: {species_url}")
    try:
        response = requests.get(species_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        species_name = get_species_name(species_url, soup)
        image_urls = []

        for img in find_image_candidates(soup):
            img_url = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not img_url:
                continue
            full_img_url = urljoin(species_url, img_url)
            if any(x in full_img_url.lower() for x in ["logo", "avatar", "gravatar"]) or full_img_url.endswith(".svg"):
                continue
            image_urls.append(full_img_url)

        return species_name, image_urls
    except Exception as e:
        print(f"Error al procesar la especie: {e}")
        return None, []


def download_images(species_name: str, image_urls, target_folder: str):
    if not image_urls:
        print("   No se encontraron imágenes válidas.")
        return

    count = 1
    saved = 0
    for img_url in image_urls:
        ext = os.path.splitext(urlparse(img_url).path)[1].lower()
        if ext not in {".jpg", ".jpeg"}:
            continue

        if saved >= 4:
            break

        try:
            img_response = requests.get(img_url, headers=HEADERS, timeout=20)
            img_response.raise_for_status()

            filename = f"{species_name}_{count}{ext}"
            filepath = os.path.join(target_folder, filename)

            with open(filepath, "wb") as handler:
                handler.write(img_response.content)

            print(f"   Guardada: {filename}")
            saved += 1
            count += 1
        except Exception as e:
            print(f"   Error descargando {img_url}: {e}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    species_links = get_species_links(START_URL)
    print(f"Se encontraron {len(species_links)} especies.")

    for species_url in species_links:
        species_name, image_urls = extract_mushroom_images(species_url)
        if species_name and image_urls:
            download_images(species_name, image_urls, OUTPUT_DIR)
        time.sleep(0.5)

    print(f"\nProceso finalizado. Imágenes guardadas en: {OUTPUT_DIR}")
