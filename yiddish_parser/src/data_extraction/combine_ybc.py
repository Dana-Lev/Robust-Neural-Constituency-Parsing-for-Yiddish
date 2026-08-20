import os
import glob

# --- CONFIGURATION ---
# UPDATED: Matches your download script's output folder
INPUT_DIR = "data/raw/ybc/books" 
OUTPUT_FILE = "data/raw/ybc/context.txt"

def combine_files():
    # 1. Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: The directory {INPUT_DIR} does not exist.")
        return

    # 2. Find all .txt files
    # The sort ensures we always combine them in the same order (200000, 200001...)
    input_files = sorted(glob.glob(f"{INPUT_DIR}/*.txt"))
    
    if not input_files:
        print(f"❌ No .txt files found in {INPUT_DIR}")
        return

    print(f"📚 Found {len(input_files)} books. Combining...")
    
    # 3. Write to the single huge context file
    # We use 'w' to overwrite any old version
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        count = 0
        total_chars = 0
        
        for fname in input_files:
            try:
                with open(fname, 'r', encoding='utf-8') as infile:
                    text = infile.read().strip()
                    # Skip empty files
                    if len(text) > 100: 
                        outfile.write(text)
                        outfile.write("\n\n") # 2 newlines between books
                        
                        count += 1
                        total_chars += len(text)
                        
                        if count % 50 == 0:
                            print(f"   ...processed {count} books so far")
            except Exception as e:
                print(f"   ⚠️ Error reading {fname}: {e}")
                
    print(f"✅ Success! Combined {count} books into: {OUTPUT_FILE}")
    print(f"   Total size: {total_chars / 1_000_000:.2f} Million characters")

if __name__ == "__main__":
    combine_files()