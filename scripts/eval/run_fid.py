import torch
from cleanfid import fid
import argparse
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--fake", required=True)
    args = parser.parse_args()

    # cleanfid sometimes expects flat directories with just images.
    # We will copy the valid images to temporary directories to ensure a clean run.
    real_tmp = "tmp_real"
    fake_tmp = "tmp_fake"
    os.makedirs(real_tmp, exist_ok=True)
    os.makedirs(fake_tmp, exist_ok=True)

    print("Copying real images...")
    real_count = 0
    for file in os.listdir(args.real):
        if file.endswith(".png"):
            shutil.copyfile(os.path.join(args.real, file), os.path.join(real_tmp, f"{real_count:05d}.png"))
            real_count += 1
            
    print("Copying fake images...")
    fake_count = 0
    for file in os.listdir(args.fake):
        if file.endswith(".png"):
            shutil.copyfile(os.path.join(args.fake, file), os.path.join(fake_tmp, f"{fake_count:05d}.png"))
            fake_count += 1

    print(f"Found {real_count} real images and {fake_count} fake images.")

    print("Calculating FID...")
    fid_score = fid.compute_fid(real_tmp, fake_tmp, device=torch.device("cpu"), num_workers=0)
    
    print("Calculating KID...")
    kid_score = fid.compute_kid(real_tmp, fake_tmp, device=torch.device("cpu"), num_workers=0)

    print(f"\n--- Results ---")
    print(f"FID: {fid_score:.4f}")
    print(f"KID: {kid_score:.4f}")

    # cleanup
    shutil.rmtree(real_tmp)
    shutil.rmtree(fake_tmp)

if __name__ == '__main__':
    main()
