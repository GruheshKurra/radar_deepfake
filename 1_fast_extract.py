#!/usr/bin/env python3
import argparse
import zipfile
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import os


def extract_chunk(args):
    zip_path, files_chunk, output_dir = args
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file_info in files_chunk:
            try:
                zf.extract(file_info, output_dir)
            except Exception as e:
                print(f"Error extracting {file_info.filename}: {e}")
    return len(files_chunk)


def main():
    parser = argparse.ArgumentParser(description='Fast parallel zip extraction')
    parser.add_argument('--zip_path', type=str, default='./data/radar-deepfake-frames.zip')
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    num_workers = args.num_workers or mp.cpu_count()
    print(f"Extracting with {num_workers} parallel workers...")

    with zipfile.ZipFile(args.zip_path, 'r') as zf:
        all_files = [info for info in zf.infolist() if not info.is_dir()]

    total_files = len(all_files)
    chunk_size = (total_files + num_workers - 1) // num_workers
    chunks = [all_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    worker_args = [(args.zip_path, chunk, args.output_dir) for chunk in chunks]

    with mp.Pool(num_workers) as pool:
        with tqdm(total=total_files, desc="Extracting files") as pbar:
            for count in pool.imap_unordered(extract_chunk, worker_args):
                pbar.update(count)

    print(f"\n✓ Extraction complete: {args.output_dir}/combined")


if __name__ == '__main__':
    main()
