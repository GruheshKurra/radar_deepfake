# ⚡ Fast Dataset Setup Guide

## Speed Comparison

| Method | Setup Time | Disk Usage | Best For |
|--------|-----------|------------|----------|
| **Train from ZIP** | 0 seconds | 13 GB | Quick experiments, limited disk |
| **Parallel Extract** | 5-10 minutes | 26 GB | Production training, maximum speed |
| Old extraction | 60+ minutes | 26 GB | ❌ Don't use |

## Option 1: Instant Training from ZIP (Recommended)

```bash
# 1. Download dataset (3-4 minutes)
./0_download_combined_dataset.sh

# 2. Start training immediately (auto-detects ZIP!)
python 2_train_model.py \
  --data_dir ./data/radar-deepfake-frames.zip \
  --batch_size 128 \
  --num_epochs 30
```

**Pros:**
- ✅ Zero extraction time
- ✅ Saves 13 GB disk space
- ✅ Multi-worker safe (fixed!)

**Cons:**
- ~2-5% slower during training (negligible)

## Option 2: Fast Parallel Extraction

For maximum training speed, extract with parallel workers:

```bash
# 1. Download dataset
./0_download_combined_dataset.sh

# 2. Fast parallel extraction (5-10 minutes on 96 cores)
python 1_fast_extract.py \
  --zip_path ./data/radar-deepfake-frames.zip \
  --num_workers 96

# 3. Train normally
python 2_train_model.py --data_dir ./data/combined
```

**Extraction Speed:**
- 96 cores: ~5-7 minutes
- 48 cores: ~10-12 minutes
- 24 cores: ~15-20 minutes
- 8 cores: ~30-40 minutes

## Unified Training Script

The main `2_train_model.py` now **auto-detects** whether you're using a ZIP file or extracted directory:

```bash
# Works with ZIP file
python 2_train_model.py --data_dir ./data/radar-deepfake-frames.zip

# Works with extracted directory
python 2_train_model.py --data_dir ./data/combined

# Auto-detects based on file extension and type!
```

**No separate scripts needed!** One script handles both modes intelligently.

## Key Fixes Made

### 1. Multi-Worker Safety
```python
# Old (BROKEN with num_workers > 0)
self._zip_file = zipfile.ZipFile(self.zip_path, 'r')

# New (SAFE with multi-worker DataLoader)
worker_info = torch.utils.data.get_worker_info()
current_worker_id = worker_info.id if worker_info else None
if self._worker_id != current_worker_id:
    # Each worker gets its own zip handle
    self._zip_file = zipfile.ZipFile(self.zip_path, 'r')
```

### 2. DataLoader Optimization
```python
DataLoader(
    dataset,
    num_workers=8,
    persistent_workers=True,  # Reuse workers between epochs
    prefetch_factor=2,        # Prefetch 2 batches per worker
    pin_memory=True           # Async GPU transfer
)
```

### 3. Parallel Extraction
Uses multiprocessing to extract chunks in parallel:
- Splits file list into equal chunks
- Each worker extracts its chunk independently
- Progress bar shows real-time status

## Troubleshooting

### "Too many open files" error
```bash
# Increase file descriptor limit
ulimit -n 65536
```

### Slow extraction on HDD
```bash
# Extract to RAM disk if you have >20 GB RAM
python 1_fast_extract.py \
  --zip_path ./data/radar-deepfake-frames.zip \
  --output_dir /dev/shm/data \
  --num_workers 32
```

### ZIP training slower than expected
- Increase `num_workers` (try 8-16)
- Enable `persistent_workers=True`
- Use SSD/NVMe storage for zip file

## Performance Tips

1. **For fastest training**: Use parallel extraction to SSD
2. **For quick experiments**: Use ZIP loading directly
3. **For RAM-rich systems**: Extract to `/dev/shm` (RAM disk)
4. **For disk-constrained**: Use ZIP loading (saves 13 GB)

## File Structure

```
data/
├── radar-deepfake-frames.zip (13 GB) - Keep for ZIP loading
└── combined/                  (13 GB extracted)
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```
