"""Download Spüler 2017 Dataset via Python with Proper Headers."""
import os
import sys
import json
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "spuler2017")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "spuler2017.rar")

API_URL = "https://api.figshare.com/v2/articles/4640686/files"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

def main():
    print(f"Querying Figshare API: {API_URL}...")
    req = urllib.request.Request(API_URL, headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as resp:
            files = json.loads(resp.read().decode())
            print(f"Found {len(files)} file(s):")
            for f in files:
                print(f"  Name: {f.get('name')}, Size: {f.get('size') / (1024*1024):.2f} MB, URL: {f.get('download_url')}")
            
            if files:
                download_url = files[0].get("download_url")
                print(f"\nDownloading from {download_url} to {OUT_FILE}...")
                dl_req = urllib.request.Request(download_url, headers=HEADERS)
                with urllib.request.urlopen(dl_req) as dl_resp, open(OUT_FILE, "wb") as out_f:
                    total = int(dl_resp.headers.get("content-length", 0))
                    downloaded = 0
                    while chunk := dl_resp.read(1024 * 1024):
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({downloaded/total*100:.1f}%)")
                        else:
                            sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB")
                        sys.stdout.flush()
                print("\n[SUCCESS] Download complete!")
    except Exception as e:
        print(f"\n[ERROR] API request failed: {e}")
        print("\nDirect browser download link:")
        print("👉 https://figshare.com/articles/dataset/A_High-Speed_Brain-Computer_Interface_BCI_Using_Dry_EEG_Electrodes/4640686")

if __name__ == "__main__":
    main()
