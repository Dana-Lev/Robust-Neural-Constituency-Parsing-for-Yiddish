import os
import sys
from nltk import Tree

# Configuration
DATA_DIR = 'data/processed/supar_ready'
FILES = ['train.txt', 'dev.txt', 'test.txt']

def is_valid_tree(tree):
    """
    Recursively checks if a tree or any of its subtrees are empty.
    An empty node (e.g., (NP )) has a length of 0 and causes Supar to crash.
    """
    # If the node itself is empty (has no children/leaves)
    if len(tree) == 0:
        return False
    
    # Recursively check children
    for child in tree:
        if isinstance(child, Tree):
            if not is_valid_tree(child):
                return False
                
    return True

def clean_file(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f" Skipping {filename} (not found)")
        return

    print(f"🧹 Scanning {filename}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    valid_lines = []
    removed_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse string to NLTK Tree
            tree = Tree.fromstring(line)
            
            # Check validity
            if is_valid_tree(tree):
                valid_lines.append(line)
            else:
                removed_count += 1
                # Optional: Print the first few bad ones to see what's wrong
                if removed_count <= 3:
                    print(f"  Removing Line {i+1} (Empty Node): {line[:50]}...")
                    
        except Exception as e:
            # If NLTK can't even parse it, it's definitely garbage
            removed_count += 1
            print(f"Removing Line {i+1} (Parse Error): {e}")

    # Write back only if we removed something
    if removed_count > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + '\n')
        print(f"Cleaned {filename}: Removed {removed_count} broken trees. Saved {len(valid_lines)} trees.")
    else:
        print(f"{filename} was already clean.")

if __name__ == "__main__":
    for f in FILES:
        clean_file(f)
