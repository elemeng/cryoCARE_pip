#!/usr/bin/env python3
"""
cryocare_batch_predict - Simple batch processing for cryoCARE
No fancy shit. Just gets the job done.
"""

import json
import subprocess
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Process cryoCARE pairs in batch. One GPU, one job.')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use (comma separated)')
    args = parser.parse_args()

    # Parse GPUs
    gpus = [g.strip() for g in args.gpus.split(',')]
    print(f"Using GPUs: {gpus}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Check pairs
    even_files = config['even']
    odd_files = config['odd']
    
    if len(even_files) != len(odd_files):
        print("ERROR: Even and odd lists don't match!")
        sys.exit(1)
    
    print(f"Processing {len(even_files)} pairs")
    
    # Process each pair
    for i, (even, odd) in enumerate(zip(even_files, odd_files)):
        gpu = gpus[i % len(gpus)]
        
        # Create temp config for this pair
        temp_config = config.copy()
        temp_config['even'] = [even]
        temp_config['odd'] = [odd]
        temp_config['gpu_id'] = [int(gpu)]
        
        temp_config_file = f'temp_config_{i}.json'
        with open(temp_config_file, 'w') as f:
            json.dump(temp_config, f)
        
        # Run it
        print(f"Pair {i+1}/{len(even_files)}: {even} + {odd} on GPU {gpu}")
        try:
            subprocess.run(['python', 'predict.py', '--config', temp_config_file], check=True)
            print("  ✓ Done")
        except subprocess.CalledProcessError:
            print("  ✗ Failed")
        
        # Clean up
        os.remove(temp_config_file)
    
    print("All done.")

if __name__ == '__main__':
    main()