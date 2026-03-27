import requests
import os


SOURCES = [
    "https://www.gutenberg.org/files/15623/15623-0.txt", 
    "https://www.gutenberg.org/cache/epub/29895/pg29895.txt" 
]

OLD_FILE = "inside_criminal_mind.txt"
NEW_FILE = "training_data.txt"

def download_and_merge():
    combined_new_text = ""
    
    for url in SOURCES:
        print(f"[INFO] Downloading: {url}")
        try:
            response = requests.get(url)
            response.encoding = 'utf-8'
            text = response.text
            
           
            start_marker = "*** START OF"
            end_marker = "*** END OF"
            
            start_idx = text.find(start_marker)
            end_idx = text.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
               
                actual_start = text.find("\n", start_idx + 50) 
                text = text[actual_start:end_idx]
            
            combined_new_text += "\n\n" + text
            print(f"[SUCCESS] Added {url.split('/')[-1]}")
            
        except Exception as e:
            print(f"[ERROR] Failed to download {url}: {e}")

    
    if os.path.exists(OLD_FILE):
        with open(OLD_FILE, 'r', encoding='utf-8') as f:
            original = f.read()
        
        final_text = original + combined_new_text
        
        with open(NEW_FILE, 'w', encoding='utf-8') as f:
            f.write(final_text)
            
        print(f"\n[COMPLETE] Created '{NEW_FILE}'")
        print(f"Total size: {len(final_text):,} characters.")
    else:
        print(f"[ERROR] Could not find {OLD_FILE}. Ensure it is in the same folder.")

if __name__ == "__main__":
    download_and_merge()
