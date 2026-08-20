import os
import re

def finalize_supar_format(input_path, output_path):
    """
    Standardizes PPCHY trees for SuPar:
    1. Wraps fragmented trees in a (TOP ...) root.
    2. Removes empty nodes like (NP-SBJ ).
    3. Ensures one tree per line.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found!")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    final_trees = []
    print(f"🔄 Processing {len(lines)} raw lines...")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 1. Clean up empty/noisy nodes created during previous steps
        # Removes things like (NP-SBJ ) or (-NONE- )
        line = re.sub(rf'\(\S+\s+\)', '', line)
        
        # 2. Ensure exactly one root by wrapping in (TOP ...)
        # This handles fragments like (IP-MAT ...) (NP-MSR ...) (PUNC ,)
        if not line.startswith("(TOP"):
            line = f"(TOP {line})"
        
        # 3. Collapse multiple spaces into one
        line = " ".join(line.split())
        
        # 4. Final Validation: Parentheses must be balanced
        if line.count('(') == line.count(')'):
            final_trees.append(line)
        else:
            # If they are unbalanced, the tree is corrupted and SuPar will crash
            continue

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_trees) + '\n')
    
    print(f"✅ Success! Created {len(final_trees)} SuPar-ready trees.")
    print(f"📍 Location: {output_path}")

if __name__ == "__main__":
    # Update these paths to match your directory
    INPUT = 'data/processed/ppchy_final_trees.txt'
    OUTPUT = 'data/processed/supar_train_ready.txt'
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    
    finalize_supar_format(INPUT, OUTPUT)