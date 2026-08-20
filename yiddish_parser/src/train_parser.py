import os
import sys
import torch
import transformers

# =========================================================
# PATCH: Inject AdamW
# =========================================================
transformers.AdamW = torch.optim.AdamW
try:
    from transformers import optimization
    if not hasattr(optimization, 'AdamW'):
        optimization.AdamW = torch.optim.AdamW
except ImportError:
    pass
# =========================================================

import supar
try:
    from supar.utils import Config
except ImportError:
    from supar.config import Config

from supar import CRFConstituencyParser

# ================= CONFIGURATION =================
# FIX: Separate the Folder from the File
OUTPUT_DIR = './output/phase3_parser_model'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'yiddish_parser.pt')

# Make the folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = 'data/processed/supar_ready'
ENCODER_PATH = './output/phase2_trained/checkpoint-6000' 
# =================================================

def freeze_recursive(module):
    frozen_count = 0
    name = module.__class__.__name__
    if 'Embeddings' in name and 'Bert' in name: 
        print(f"      Found Candidate: {name}")
        for param in module.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        return frozen_count

    for child in module.children():
        frozen_count += freeze_recursive(child)
    return frozen_count

def apply_freeze_patch():
    print("\n APPLYING FREEZE PATCH...")
    OriginalModel = CRFConstituencyParser.MODEL
    
    class FrozenModel(OriginalModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print(f"  INTERCEPTED: Searching for embeddings in {self.__class__.__name__}...")
            
            count = 0
            if hasattr(self, 'encoder'):
                count = freeze_recursive(self.encoder)
            
            if count == 0:
                print("Encoder scan empty. Trying generic scan...")
                for m in self.modules():
                    if 'Embeddings' in m.__class__.__name__:
                         for p in m.parameters():
                             p.requires_grad = False
                             count += p.numel()

            if count > 0:
                print(f"SUCCESS: Frozen {count/1e6:.2f}M parameters.")
            else:
                print("ERROR: Could not find any embeddings to freeze!")

    CRFConstituencyParser.MODEL = FrozenModel
    print("Patch Applied.\n")

def train():
    apply_freeze_patch()

    args = {
        # 1. Paths
        'train': os.path.join(DATA_DIR, 'train.txt'),
        'dev': os.path.join(DATA_DIR, 'dev.txt'),
        'test': os.path.join(DATA_DIR, 'test.txt'),
        
        #FIX: Save to the FILE, not the FOLDER
        'path': MODEL_FILE,
        
        'mode': 'train',
        'build': True,
        'checkpoint': False, 

        # 2. Encoder
        'encoder': 'bert',
        'bert': ENCODER_PATH,
        'finetune': False,
        'n_bert_layers': 4,
        'bert_pooling': 'mean',
        'mix_dropout': 0.0,
        
        # 3. Model Architecture
        'feat': ['char'],
        'n_embed': 100,
        'n_char_embed': 50,
        'n_char_hidden': 100,
        'n_feat_embed': 100,
        'n_encoder_hidden': 800,
        'n_encoder_layers': 3,
        'encoder_dropout': 0.33,
        'n_span_mlp': 500,
        'n_label_mlp': 100,
        'mlp_dropout': 0.33,
        'embed_dropout': 0.33,
        
        # 4. Training
        'lr': 5e-5,
        'lr_rate': 20,
        'batch_size': 2000,
        'epochs': 100,
        'patience': 10,
        'update_steps': 1,
        'warmup': 0.1,
        'clip': 5.0,
        'decay': 0.75,
        'decay_steps': 5000,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'eps': 1e-8,
        'mu': 0.9,
        'nu': 0.9,
        
        # 5. Eval
        'structure': 'joint',
        'mbr': True,
        'delete': {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
        'equal': {'ADVP': 'PRT'},
        
        # 6. System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 1,
        'amp': False,
        'cache': False,
        'verbose': True,
        'punct': False,
        'buckets': 32,
        'workers': 0 
    }
    
    print(f"Starting Training for: {ENCODER_PATH}")
    print(f"Saving model to: {MODEL_FILE}")
    
    config = Config(**args)
    print("   -> Building Parser instance...")
    parser = CRFConstituencyParser.build(**config)
    
    print("   -> Starting Training Loop...")
    parser.train(**config)
    print("Training Complete.")

if __name__ == "__main__":
    train()
