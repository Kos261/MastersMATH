import os
import urllib.request

BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/"

FILES = {
    # Efemerydy planet
    "generic_kernels/spk/planets/de442s.bsp": "de442s.bsp",
    "generic_kernels/lsk/naif0012.tls":       "naif0012.tls",
    # https: // ssd.jpl.nasa.gov / ftp / eph / small_bodies / asteroids_de441 / sb441 - n16.bsp
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    for remote_path, local_name in FILES.items():
        url = BASE_URL + remote_path
        dest = os.path.join(DATA_DIR, local_name)

        if os.path.exists(dest):
            print(f"[skip] {local_name} już istnieje")
            continue

        print(f"[download] {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[ok] {local_name}")
        except Exception as e:
            print(f"[error] Nie udało się pobrać {local_name}: {e}")


if __name__ == "__main__":
    download_all()