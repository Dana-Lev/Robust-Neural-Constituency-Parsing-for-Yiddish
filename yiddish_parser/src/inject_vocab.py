import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from deepfocus import FOCUS

# --- CONFIGURATION ---
ORIGINAL_MODEL = "skulick/xlmb-ybc-ck05"
VOCAB_FILE = "./data/processed/yiddish_vocab.txt"
OUTPUT_DIR = "./output/phase1_focus_model"
TARGET_NEW_TOKENS = 2000

# 📍 DIRECT PATHS (The simple way)
# 1. Your 4.5GB FastText Brain
FASTTEXT_PATH = "/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman/fasttext_models/cc.yi.300.bin"
# 2. Your JSONL Data (Created in Step 1)
DATA_PATH = "./data/processed/ybc_focus.jsonl" 

# 🛡️ Cache Safety
CACHE_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman/cache"

def run_official_focus():
    print(f"🚀 Starting FOCUS Injection (Standard Mode)...")
    
    # 1. Validation
    if not os.path.exists(FASTTEXT_PATH):
        print(f"❌ Error: FastText file missing at {FASTTEXT_PATH}")
        return
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: JSONL data missing at {DATA_PATH}. Run the conversion one-liner first!")
        return

    # 2. Load Base Model
    print(f"   Loading base model: {ORIGINAL_MODEL}")
    source_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL, cache_dir=CACHE_DIR)
    source_model = AutoModelForMaskedLM.from_pretrained(ORIGINAL_MODEL, cache_dir=CACHE_DIR)

    # 3. Create Target Tokenizer
    print("   Preparing new vocabulary...")
    target_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL, cache_dir=CACHE_DIR)
    # This file contains the 2,000 common words from Jochre we found earlier
    with open(VOCAB_FILE, 'r') as f:
        candidates = [line.strip() for line in f if line.strip()]
    
    tokens_to_add = []
    for word in candidates:
        if len(tokens_to_add) >= TARGET_NEW_TOKENS: break
        if len(source_tokenizer.tokenize(word)) > 1:
            tokens_to_add.append(word)
            
    print(f"   Adding {len(tokens_to_add)} new tokens...")
    target_tokenizer.add_tokens(tokens_to_add)

    # 4. RUN FOCUS ⚗️
    # We pass the paths directly, exactly like the README says.
    print("   ⚗️  Running FOCUS (Using Local FastText + YBC Data)...")
    
    source_embeddings = source_model.get_input_embeddings().weight
    
    target_embeddings = FOCUS(
        source_embeddings=source_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        target_training_data_path=DATA_PATH,  # <--- Your Data
        fasttext_model_path=FASTTEXT_PATH     # <--- Your 4.5GB File
    )

    # 5. Save
    print("   💾 Applying new embeddings...")
    source_model.resize_token_embeddings(len(target_tokenizer))
    source_model.get_input_embeddings().weight.data = target_embeddings
    
    source_model.save_pretrained(OUTPUT_DIR)
    target_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ SUCCESS! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_official_focus()