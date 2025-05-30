#!/usr/bin/env python3
import sys
import os
import shutil
from huggingface_hub import snapshot_download
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Download LLaVA-Pretrain dataset and organize files."
    )
    parser.add_argument(
        "--repo-id", default="liuhaotian/LLaVA-Pretrain",
        help="HuggingFace dataset repo"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Root folder to place downloaded data"
    )
    args = parser.parse_args()

    repo_id = args.repo_id
    root_out = os.path.abspath(args.output_dir)
    tmp_dir = os.path.join(root_out, "LLaVA-Pretrain")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Downloading {repo_id} to {tmp_dir} ...")
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=tmp_dir,
            local_dir_use_symlinks=False
        )
        print(f"Download complete. Files in: {download_path}")
    except Exception as e:
        print(f"Error during download: {e}", file=sys.stderr)
        sys.exit(1)

    # Giải nén images.zip
    zip_path = os.path.join(tmp_dir, "images.zip")
    images_out = os.path.join(tmp_dir, "images")
    if os.path.isfile(zip_path):
        print(f"Unzipping {zip_path} ...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(images_out)
        os.remove(zip_path)
        print(f"Extracted images to: {images_out}")
    else:
        print("Warning: images.zip not found!")

    # Di chuyển 2 file JSON và thư mục images lên output-dir
    for fname in os.listdir(tmp_dir):
        src = os.path.join(tmp_dir, fname)
        if fname.endswith(".json") or fname == "images":
            dst = os.path.join(root_out, fname)
            print(f"Moving {src} -> {dst}")
            if os.path.exists(dst):
                # xóa nếu đã tồn tại (đảm bảo clean)
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)

    # Xóa thư mục tạm
    print(f"Cleaning up temporary folder {tmp_dir}")
    shutil.rmtree(tmp_dir)

    # Kiểm tra size tập images
    check_zip = os.path.join(root_out, "images")
    if os.path.isdir(check_zip):
        total_bytes = 0
        for dirpath, _, filenames in os.walk(check_zip):
            for f in filenames:
                total_bytes += os.path.getsize(os.path.join(dirpath, f))
        gb = total_bytes / (1024 ** 3)
        print(f"Total size of images folder: {gb:.2f} GB")

if __name__ == "__main__":
    main()
