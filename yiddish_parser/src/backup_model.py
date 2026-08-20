import os
from huggingface_hub import login # <--- Import login tool

# --- 1. CONFIGURATION ---
HF_USERNAME = "Omriboiman"
LOCAL_DIR = "./output/phase2_trained/checkpoint-6000"
REPO_NAME = f"{HF_USERNAME}/yiddish-xlm-roberta-DAPT-6000"

# --- 2. PASTE TOKEN HERE (Solves the 401 Error) ---
# Replace the text inside the quotes with your actual token starting with hf_
MY_TOKEN = "" 

# --- 3. FORCE CACHE TO BIG DRIVE ---
CACHE_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/omriboiman/cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
print(f"🔧 Forced Hugging Face cache to: {CACHE_DIR}")

# --- 4. LOGIN & IMPORT ---
if MY_TOKEN == "PASTE_YOUR_HF_TOKEN_HERE":
    print("❌ ERROR: You didn't paste your token in the script!")
    exit()

print("🔑 Logging in...")
login(token=MY_TOKEN) # <--- Authenticates directly

from transformers import AutoModelForMaskedLM, AutoTokenizer

def backup():
    print(f"🚀 Preparing to backup model from: {LOCAL_DIR}")
    
    if not os.path.exists(LOCAL_DIR):
        print(f"❌ Error: Could not find {LOCAL_DIR}")
        return

    try:
        print("   Loading model into memory...")
        model = AutoModelForMaskedLM.from_pretrained(LOCAL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"☁️  Uploading to Hugging Face: {REPO_NAME}...")
    
    try:
        model.push_to_hub(REPO_NAME, private=True)
        tokenizer.push_to_hub(REPO_NAME, private=True)
        print("✅ Backup Successful!")
        print(f"   View it here: https://huggingface.co/{REPO_NAME}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    backup()
