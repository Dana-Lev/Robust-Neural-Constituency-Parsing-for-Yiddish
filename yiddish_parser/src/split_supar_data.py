import random
import os

def split_data(input_file, output_dir, train_p=0.9, dev_p=0.05):
    # Ensure the input exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    # Create the clean directory
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    # Shuffle for randomness (Seed 42 for reproducibility)
    random.seed(42)
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * train_p)
    dev_end = int(total * (train_p + dev_p))
    
    train_data = lines[:train_end]
    dev_data = lines[train_end:dev_end]
    test_data = lines[dev_end:]
    
    # Save files to the new folder
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data) + '\n')
    with open(os.path.join(output_dir, 'dev.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(dev_data) + '\n')
    with open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_data) + '\n')
    
    print(f"📂 Files organized in: {output_dir}")
    print(f"✅ Split Summary: Train({len(train_data)}), Dev({len(dev_data)}), Test({len(test_data)})")

if __name__ == "__main__":
    split_data('data/processed/supar_train_ready.txt', 'data/processed/supar_ready')