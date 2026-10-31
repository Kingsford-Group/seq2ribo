import pickle
import hashlib
import argparse
import os
import sys

def get_hash(seq_str):
    # Ensure it's a string
    if not isinstance(seq_str, str):
        # Try to convert bytes or other formats if necessary, or skip
        seq_str = str(seq_str)
    
    # Canonicalize: Uppercase and U->T
    s = seq_str.upper().replace("U", "T")
    sig = "CODS|" + s
    return hashlib.md5(sig.encode("utf-8")).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Filter test data based on training sequence overlap.")
    parser.add_argument('--train_pkl', required=True, help="Path to train_data.pkl")
    parser.add_argument('--test_pkl', required=True, help="Path to test_data.pkl")
    parser.add_argument('--output_pkl', required=True, help="Path to output test_data_filtered.pkl")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.train_pkl):
        print(f"Error: Train PKL not found: {args.train_pkl}")
        sys.exit(1)
    if not os.path.exists(args.test_pkl):
        print(f"Error: Test PKL not found: {args.test_pkl}")
        sys.exit(1)

    print(f"Loading Training Data from {args.train_pkl}...")
    with open(args.train_pkl, 'rb') as f:
        train_data = pickle.load(f)
    
    print("Building Training Hash Set...")
    train_hashes = set()
    for i, item in enumerate(train_data):
        # item[1] is sequence
        if len(item) > 1:
            h = get_hash(item[1])
            train_hashes.add(h)
            
    print(f"Unique Training Hashes: {len(train_hashes)}")
    
    print(f"Loading Test Data from {args.test_pkl}...")
    with open(args.test_pkl, 'rb') as f:
        test_data = pickle.load(f)
        
    filtered_test = []
    removed_count = 0
    
    print("Filtering Test Data...")
    for item in test_data:
        if len(item) > 1:
            h = get_hash(item[1])
            if h in train_hashes:
                removed_count += 1
            else:
                filtered_test.append(item)
        else:
            # If structure is weird, keep or drop? Assuming drop or keep. 
            # If no sequence, can't overlap. Keep.
            filtered_test.append(item)
            
    print(f"Original Test Size: {len(test_data)}")
    print(f"Removed Overlapping: {removed_count}")
    print(f"Filtered Test Size: {len(filtered_test)}")
    
    print(f"Saving to {args.output_pkl}...")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(filtered_test, f)
    print("Done.")

if __name__ == "__main__":
    main()
