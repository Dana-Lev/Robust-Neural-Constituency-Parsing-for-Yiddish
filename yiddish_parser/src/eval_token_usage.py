import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm
import collections
import os

# --- CONFIGURATION ---
MODEL_PATH = "./output/phase2_trained/checkpoint-6000"
TEST_FILE = "./data/processed/jochre_test_combined.txt" 
VOCAB_FILE = "./data/processed/yiddish_vocab.txt"

# XLM-R Base Vocabulary Size (The cutoff line)
ORIGINAL_VOCAB_SIZE = 250002

def evaluate_usage():
    print(f"🕵️‍♂️ Loading Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    model.eval()
    
    # 1. Load Vocabulary and Separate OLD vs NEW
    print(f"📖 Analyzing vocabulary...")
    with open(VOCAB_FILE, 'r') as f:
        target_words = set(line.strip() for line in f if line.strip())
    
    native_ids = set()
    injected_ids = set()
    
    for word in target_words:
        # Get the ID
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            token_id = ids[0]
            if token_id < ORIGINAL_VOCAB_SIZE:
                native_ids.add(token_id)
            else:
                injected_ids.add(token_id)
            
    print(f"📊 Vocabulary Breakdown:")
    print(f"   - Native Yiddish Words (Old):  {len(native_ids)}")
    print(f"   - Injected Yiddish Words (New): {len(injected_ids)}")
    print(f"   ---------------------------------------")
    print(f"   - Total Tracking:              {len(native_ids) + len(injected_ids)}")

    # 2. Prepare Data
    if not os.path.exists(TEST_FILE):
        print(f"❌ Error: {TEST_FILE} not found.")
        return

    print(f"📚 Preparing test data...")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=TEST_FILE,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

    # 3. Run Evaluation
    print("🚀 Measuring token emissions...")
    total_predictions = 0
    
    # Counters
    native_emissions = 0
    injected_emissions = 0
    
    token_counts = collections.Counter()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using Device: {device}")
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            mask = labels != -100
            valid_preds = predictions[mask]
            
            for token_id in valid_preds.tolist():
                total_predictions += 1
                token_counts[token_id] += 1
                
                if token_id in native_ids:
                    native_emissions += 1
                elif token_id in injected_ids:
                    injected_emissions += 1

    # 4. Report Results (THE REAL TRUTH)
    print("\n" + "="*50)
    print("📊 ZOMBIE TOKEN REPORT (Strict Separation)")
    print("="*50)
    
    # A. Native Words (Control Group)
    unique_native_used = sum(1 for tid in native_ids if token_counts[tid] > 0)
    native_coverage = (unique_native_used / len(native_ids)) * 100 if len(native_ids) > 0 else 0
    
    print(f"🟢 NATIVE Words (Already known):")
    print(f"   - Emitted Total:  {native_emissions}")
    print(f"   - Unique Used:    {unique_native_used} / {len(native_ids)} ({native_coverage:.2f}%)")
    
    print("-" * 50)
    
    # B. Injected Words (Test Group)
    unique_injected_used = sum(1 for tid in injected_ids if token_counts[tid] > 0)
    injected_coverage = (unique_injected_used / len(injected_ids)) * 100 if len(injected_ids) > 0 else 0
    
    print(f"🔴 INJECTED Words (The New Ones):")
    print(f"   - Emitted Total:  {injected_emissions}")
    print(f"   - Unique Used:    {unique_injected_used} / {len(injected_ids)} ({injected_coverage:.2f}%)")
    
    print("="*50)
    
    if injected_coverage < 5:
        print("🧟 DIAGNOSIS: ZOMBIE OUTBREAK.")
        print("   The model is using the Native words but IGNORING the Injected ones.")
    else:
        print("🎉 DIAGNOSIS: SUCCESS.")
        print("   The model is actively using the new Injected words!")

if __name__ == "__main__":
    evaluate_usage()