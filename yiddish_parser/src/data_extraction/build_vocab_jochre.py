import os
import glob
import re
from collections import Counter

# --- CONFIGURATION ---
# STRICTLY use Jochre for the vocabulary definition (Gold Standard)
JOCHRE_DIR = "./data/raw/jochre" 
OUTPUT_FILE = "./data/processed/yiddish_vocab.txt"

# Regex to find words containing Hebrew characters (Yiddish/Hebrew script)
# \u0590-\u05FF is the Unicode range for Hebrew
WORD_PATTERN = re.compile(r'[\u0590-\u05FF]+')

def get_jochre_files():
    """Recursively finds all text files in the Jochre corpus."""
    # Jochre structure usually contains .txt files in subfolders
    files = glob.glob(f"{JOCHRE_DIR}/**/*.txt", recursive=True)
    return files

def build_vocabulary():
    print("📖 Starting Vocabulary Extraction (JOCHRE ONLY)...")
    files = get_jochre_files()
    
    if not files:
        print("❌ Error: No files found in Jochre directory!")
        print(f"   Checked: {JOCHRE_DIR}")
        return

    print(f"   Found {len(files)} files to scan.")

    word_counts = Counter()
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                # Find all Yiddish words
                words = WORD_PATTERN.findall(text)
                word_counts.update(words)
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")

    # --- FILTERING ---
    # Since Jochre is high quality, we can trust words that appear even a few times.
    # We want the most frequent words that might be MISSING from the original model.
    MIN_FREQUENCY = 3 
    
    sorted_vocab = [word for word, count in word_counts.most_common() if count >= MIN_FREQUENCY]
    
    print(f"   Total unique words found: {len(word_counts)}")
    print(f"   Words appearing >={MIN_FREQUENCY} times: {len(sorted_vocab)}")
    
    # --- SAVE ---
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_vocab))
        
    print(f"✅ Saved Gold Standard vocabulary to: {OUTPUT_FILE}")
    print(f"   Top 10 words: {sorted_vocab[:10]}")

if __name__ == "__main__":
    build_vocabulary()