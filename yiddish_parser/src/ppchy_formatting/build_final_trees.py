import json
import os
import re

def clean_and_replace(tree_str, leaves):
    # 1. Skip non-tree lines (CODE blocks)
    if "CODE" in tree_str and not leaves:
        return None
    
    # 2. Sort leaves by 'start' position to be safe
    leaves = sorted(leaves, key=lambda x: x['start'])
    
    # 3. Replace Romanized words with Yiddish script
    # We look for the Romanized word at the end of a bracket: (POS word)
    for leaf in leaves:
        rom_word = leaf['rom']
        yid_word = leaf['yid']
        # Use regex to find the word specifically as a leaf node
        # This prevents accidental partial replacements
        tree_str = re.sub(rf'\((\S+)\s+{re.escape(rom_word)}\)', rf'(\1 {yid_word})', tree_str)
    
    # 4. Remove SuPar-unfriendly noise
    tree_str = re.sub(r'\(CODE .*?\)', '', tree_str)
    tree_str = re.sub(r'\(ID .*?\)', '', tree_str)
    tree_str = re.sub(r'\(-NONE- .*?\)', '', tree_str)
    
    # 5. Collapse extra spaces
    tree_str = " ".join(tree_str.split())
    
    # Final check: if the tree is empty or just a fragment, skip it
    if tree_str.count('(') < 2:
        return None
        
    return tree_str

def main():
    json_dir = "data/raw/ppchyprep/out/data/json"
    output_path = "data/processed/ppchy_final_trees.txt"
    
    all_trees = []
    
    for filename in sorted(os.listdir(json_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), 'r') as f:
                data = json.load(f)
                for entry in data:
                    final_tree = clean_and_replace(entry['tree'], entry['leaves'])
                    if final_tree:
                        all_trees.append(final_tree)
    
    with open(output_path, 'w') as f:
        for tree in all_trees:
            f.write(tree + "\n")
    
    print(f"✅ Created {len(all_trees)} Hebrew-script trees in {output_path}")

if __name__ == "__main__":
    main()