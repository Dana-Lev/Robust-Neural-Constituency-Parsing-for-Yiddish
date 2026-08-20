import argparse
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
    parser = argparse.ArgumentParser(
        description="Convert ppchyprep JSON output into Hebrew-script bracketed trees.")
    parser.add_argument("--json-dir", default="data/raw/ppchyprep/out/data/json")
    parser.add_argument("--output", default="data/processed/ppchy_final_trees.txt")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Restrict to PPCHY components whose filename contains any of these "
             "strings, e.g. --only hirshbein olsvanger to follow Kulick et al. "
             "(2022). Default: every .json file in the directory. Whatever you "
             "choose here must match what the report claims.")
    args = parser.parse_args()

    json_dir = args.json_dir
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_trees = []
    used_files = []

    for filename in sorted(os.listdir(json_dir)):
        if args.only and not any(key.lower() in filename.lower() for key in args.only):
            continue
        if filename.endswith(".json"):
            used_files.append(filename)
            with open(os.path.join(json_dir, filename), 'r') as f:
                data = json.load(f)
                for entry in data:
                    final_tree = clean_and_replace(entry['tree'], entry['leaves'])
                    if final_tree:
                        all_trees.append(final_tree)
    
    with open(output_path, 'w') as f:
        for tree in all_trees:
            f.write(tree + "\n")
    
    print(f"Created {len(all_trees)} Hebrew-script trees in {output_path}")
    print(f"Source files used ({len(used_files)}): {', '.join(used_files)}")

if __name__ == "__main__":
    main()