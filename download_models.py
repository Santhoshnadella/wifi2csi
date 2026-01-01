import gdown
import os

def download_weights():
    # URL from the paper/repo
    url = "https://drive.google.com/drive/folders/1U9hMGtMoQWgP_Obi5k1Vts_WbEHi85Vi?usp=sharing"
    output_dir = "weights"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading pretrained models from {url}...")
    print("note: This requires 'gdown' installed (pip install gdown)")
    
    try:
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Please manually download the folder from the URL above and place it in the 'weights' directory.")

if __name__ == "__main__":
    download_weights()
