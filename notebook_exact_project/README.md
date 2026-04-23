# Retina blood vessel segmentation (Notebook-exact)

This project is a **line-by-line faithful refactor** of the uploaded notebook
`retina-blood-vessel-segmentation-pytorch.ipynb`.

It keeps the same:
- Dataset reading (OpenCV, normalization, tensor layout)
- U-Net architecture
- Loss (Dice + BCE) implementation
- Training loop (no scheduler step, checkpoint on best val loss)
- Inference + metrics (sklearn) and result montage saving

## Dataset layout (same as notebook)

Default paths in the notebook were:

- `/kaggle/input/retina-blood-vessel/Data/train/image/*`
- `/kaggle/input/retina-blood-vessel/Data/train/mask/*`
- `/kaggle/input/retina-blood-vessel/Data/test/image/*`
- `/kaggle/input/retina-blood-vessel/Data/test/mask/*`

On your machine/server, set `--data_root` to the folder that contains `Data/`:

```
<data_root>/Data/train/image
<data_root>/Data/train/mask
<data_root>/Data/test/image
<data_root>/Data/test/mask
```

## Train

```bash
python train.py --data_root /path/to/retina-blood-vessel
```

Outputs:
- checkpoint: `files/checkpoint.pth`

## Inference / evaluation

```bash
python infer_quick.py --data_root /path/to/retina-blood-vessel --ckpt files/checkpoint.pth
```

Outputs:
- `results/*.png` (image | GT | prediction montage)
- prints Jaccard/F1/Recall/Precision/Acc and FPS
