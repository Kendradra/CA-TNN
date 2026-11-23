# CA-TNN

Code for *Channel-Aware Tensor Nuclear Norm: A Self-Supervised Approach for HSI Inpainting*.

The poster summarizes the method and the experiments. The sections below describe the released demo.

![ICIP 2026 poster](poster_icip2026.png)

## Contents

| Path | Description |
| --- | --- |
| `CA-TNN_demo.ipynb` | Self-supervised training on Pavia Centre with a 95% missing rate |
| `dataset/Pavia-MR95/` | Ground truth, random mask, and observed cube for that case |
| `poster_icip2026.png` | Poster shown above |

The notebook is the full training entry point. It builds the model, runs Adam, and writes the checkpoint together with the curves.

## Environment

The notebook was written for Python 3.10 and a CUDA GPU. It places every tensor on `torch.cuda.FloatTensor`.

Install:

- PyTorch with CUDA
- NumPy
- SciPy
- scikit-image
- Matplotlib
- pandas

## Demo data

`dataset/Pavia-MR95/` holds one Pavia Centre cube of size `200 × 200 × 80` (height, width, bands), stored as `float64` NumPy arrays. Pixel values of the complete cube lie in `[0, 1]`.

| File | Contents |
| --- | --- |
| `Pavia_gtruth.npy` | Complete cube |
| `Pavia_mask.npy` | `1` on observed entries, `0` on missing entries. About 5% of the entries are observed |
| `Pavia_sampled.npy` | Observed cube: kept pixels where the mask is `1`, and `0` elsewhere |

To run another cube, replace these three arrays and set `data_set`, `case`, and `folder` in the data cell. All three arrays must share the shape `(height, width, bands)`.

## Run

Open `CA-TNN_demo.ipynb` and run the cells from top to bottom. Training prints the iteration, the elapsed time, and the mean per-band PSNR every 40 steps. PSNR is the average of `skimage.metrics.peak_signal_noise_ratio` over bands.

When the loop finishes, the notebook creates a folder

`CA-TNN_Pavia_MR95_<YYYY-MM-DD HH.MM>/`

with:

| File | Contents |
| --- | --- |
| `CA-TNN_Pavia_MR95.pth` | Weights of the iterate with the highest mean per-band PSNR |
| `Pavia_MR95_CNN_FC_expSE_slice.npy` | Restored cube at that iterate, shape `(200, 200, 80)` |
| `CA-TNN_Pavia_MR95_<time>.csv` | Learning rate, `beta1`, `lambda`, transform settings, iteration count, wall time, final PSNR, and best PSNR |
| `Training Loss1.png`, `Training Loss2.png` | Loss over all iterations, and over the last four fifths |
| `Training PSNR1.png`, `Training PSNR2.png` | The same two windows for PSNR |
| `nuc_.png` | Nuclear-norm term over iterations |

## Citation

```bibtex
@inproceedings{li2026catnn,
  title={Channel-Aware Tensor Nuclear Norm: A Self-Supervised Approach for {HSI} Inpainting},
  author={Li, Yunshan and Wang, Qianqian and Gong, Wenwu and Yang, Lili},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2026}
}
```
