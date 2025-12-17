import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_data(raw_dir, output_dir, train_ratio, val_ratio, test_ratio, seed):
    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    classes = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    for cls in classes:
        cls_raw = os.path.join(raw_dir, cls)
        imgs = [
            f for f in os.listdir(cls_raw)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # First split off test set
        train_val, test = train_test_split(
            imgs, test_size=test_ratio, random_state=seed
        )
        # Then split train vs val
        val_size_rel = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_size_rel, random_state=seed
        )

        for split_name, split_files in zip(
            ['train', 'val', 'test'], [train, val, test]
        ):
            out_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for fname in split_files:
                src_path = os.path.join(cls_raw, fname)
                dst_path = os.path.join(out_dir, fname)
                shutil.copy2(src_path, dst_path)

        print(f"[{cls}] → train: {len(train)} | val: {len(val)} | test: {len(test)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split ECG images into train/val/test"
    )
    parser.add_argument(
        "--raw_dir", type=str, default="data/raw",
        help="Directory with raw class-subfolders"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Root dir for train/val/test folders"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
        help="Fraction of images for training"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="Fraction of images for validation"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15,
        help="Fraction of images for testing"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    split_data(
        args.raw_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
