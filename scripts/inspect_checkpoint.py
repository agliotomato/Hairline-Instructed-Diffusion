import torch
import sys
from collections import defaultdict

def inspect(path):
    try:
        state_dict = torch.load(path, map_location='cpu')
        print(f"Loaded {path}")
        print(f"Total Keys: {len(state_dict.keys())}")
        
        prefixes = defaultdict(int)
        for k in state_dict.keys():
            prefix = k.split('.')[0]
            prefixes[prefix] += 1
            
        print("\nKey Prefixes:")
        for p, count in prefixes.items():
            print(f"{p}: {count}")
            
        print("\nKeys not starting with 'cp' or 'ffm':")
        other_keys = [k for k in state_dict.keys() if not k.startswith('cp.') and not k.startswith('ffm.')]
        for k in other_keys:
            print(k)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect(sys.argv[1])
