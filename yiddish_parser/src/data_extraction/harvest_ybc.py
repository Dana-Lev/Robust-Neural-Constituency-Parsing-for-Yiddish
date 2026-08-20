import os
import requests
import time


# --- CONFIGURATION ---
TARGET_COUNT = 400       # We want 400 books
START_ID = 200000        # The ID to start counting from (Public domain starts here)
OUTPUT_FOLDER = "data/raw/ybc/books"

def download_books():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    current_id_num = START_ID
    downloaded_count = 0
    
    print(f"🚀 Starting mission: Download {TARGET_COUNT} Yiddish books...")
    print(f"📂 Saving to folder: {OUTPUT_FOLDER}\n")

    # Loop until we have 400 books
    while downloaded_count < TARGET_COUNT:
        # Construct the ID (e.g., "nybc200001")
        book_id = f"nybc{current_id_num}"
        
        # Try the standard text file URL
        # Note: We prioritize '_djvu.txt' as it's the standard OCR layer
        url = f"https://archive.org/download/{book_id}/{book_id}_djvu.txt"
        
        try:
            # We use stream=True to avoid downloading the whole file if header check fails
            # But Archive.org often requires a GET to check existence properly
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                # We found a valid text file!
                file_path = os.path.join(OUTPUT_FOLDER, f"{book_id}.txt")
                
                # Check if file is essentially empty (sometimes they exist but are 0 bytes)
                if len(response.content) > 1000: 
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    
                    downloaded_count += 1
                    print(f"✅ [{downloaded_count}/{TARGET_COUNT}] Saved: {book_id}")
                else:
                    print(f"⚠️  Skipped {book_id}: File too small (probably empty)")
            else:
                # 404 or 403 means book is missing or locked
                # print(f"❌ Skipped {book_id}: Not found or restricted") 
                pass

        except Exception as e:
            print(f"⚠️  Error checking {book_id}: {e}")
        
        # Move to the next ID number
        current_id_num += 1
        
        # Be polite to the server (prevents getting banned)
        if downloaded_count % 10 == 0:
            time.sleep(0.5)

    print(f"\n🎉 Mission Accomplished! You have {TARGET_COUNT} books.")
    print("Next step: Combine them into one file for training.")

if __name__ == "__main__":
    download_books()