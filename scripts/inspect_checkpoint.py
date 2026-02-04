import torch
import argparse
import os

def inspect_checkpoint(args):
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        print(f"Error: File not found at {model_path}")
        return

    print(f"Loading checkpoint from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    print("\nXXX Checkpoint Inspection Report XXX")
    print(f"Type of checkpoint object: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"\nTop-level keys found: {list(checkpoint.keys())}")
        
        for key, value in checkpoint.items():
            print(f"\n--- Key: '{key}' ---")
            print(f"  Type: {type(value)}")
            
            if isinstance(value, dict):
                print(f"  Number of items: {len(value)}")
                # Print first few keys as a sample
                keys_sample = list(value.keys())[:10]
                print(f"  Sample keys: {keys_sample}")
                if len(value) > 10:
                    print("  ...")
            
            elif isinstance(value, torch.Tensor):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
            
            elif isinstance(value, (list, tuple)):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  Type of first element: {type(value[0])}")
            
            else:
                print(f"  Value: {value}")
                
        # Special check for 'model_state_dict' or 'model' usually found in RL checkpoints
        if 'model_state_dict' in checkpoint:
            print("\n--- Detailed Model Architecture (from model_state_dict) ---")
            # Analyze layers
            layers = set()
            for key in checkpoint['model_state_dict'].keys():
                layer_name = key.split('.')[0]
                layers.add(layer_name)
            print(f"  Inferred top-level modules: {layers}")

    else:
        # If it's a raw JIT export or a straight model save
        print("\nThis might be a JIT trace/script or a raw model object, not a state_dict dictionary.")
        print(checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents of a PyTorch checkpoint (.pt) file.")
    parser.add_argument("model_path", type=str, help="Path to the .pt file")
    args = parser.parse_args()
    inspect_checkpoint(args)
