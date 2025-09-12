# Donkey3DGS – Video to 3D Gaussian Splatting Pipeline

This project demonstrates how to reconstruct a **3D Gaussian Splatting (3DGS)** model from a video of the UGV (Unmanned Ground Vehicle) using **COLMAP** for photogrammetry and the [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) repo for training and rendering.

---

## 📦 Requirements

- **GPU**: NVIDIA RTX 3090 Ti (CUDA 12.x recommended)
- **OS**: Ubuntu 22.04+
- **Dependencies**:
  - `ffmpeg` (for video → frames)
  - `COLMAP` (photogrammetry)
  - `Miniconda` (Python env)
  - NVIDIA CUDA drivers (tested with CUDA 12.1)

---

## 🔧 Environment Setup

### 1. Make a workspace & clone the repo (with submodules)
```bash
mkdir -p ~/Donkey3DGS && cd ~/Donkey3DGS
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
```
(If you already cloned without ``` --recursive```, do this:)

```bash
cd ~/Donkey3DGS/gaussian-splatting
git submodule sync --recursive
git submodule update --init --recursive --depth 1
```

### 2. Install Miniconda (skip if you already have it)
```bash
cd ~/Donkey3DGS
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

```

### 3. Create & activate a clean environment
```bash
conda create -n gs-env python=3.10 -y
conda activate gs-env
```

### 4. Install PyTorch (with CUDA 12.1 wheels)
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 5. Install system build tools + Python dependencies (OpenCV)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libgl1 colmap ffmpeg sqlite3

pip install -U pip setuptools wheel cmake ninja
pip install opencv-python-headless matplotlib scipy tqdm plyfile tensorboard
```
If ```colmap``` isn’t found by apt on your OS image, you can install via conda:
```conda install -c conda-forge colmap -y```

### 6. Build the CUDA submodules (fixes the “No module named torch” error)
```bash
cd ~/Donkey3DGS/gaussian-splatting

# Optional: set your GPU arch (RTX 30xx = 8.6, RTX 40xx = 8.9, GTX 1080 = 6.1)
export TORCH_CUDA_ARCH_LIST="8.6"
export TCNN_CUDA_ARCHITECTURES=86

# Some systems need explicit compilers:
export CC=gcc
export CXX=g++

# IMPORTANT: disable PEP517 build isolation so Torch is visible during build
export PIP_USE_PEP517=0

pip install --no-build-isolation --no-deps submodules/diff-gaussian-rasterization
pip install --no-build-isolation --no-deps submodules/simple-knn
```

### 7. Sanity check (CUDA + extensions)
```bash
python - <<'PY'
import importlib.util, torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
for m in ["diff_gaussian_rasterization", "simple_knn._C"]:
    print(f"{m} importable:", importlib.util.find_spec(m) is not None)
PY
```
Expected: `CUDA available: True` and both modules `importable: True`.


---

## 🎥 Pipeline

### 1. Extract frames from video (e.g. 5 frames per second)
```bash
# Put your video here first:
# e.g., scp my.mov zobaerdnn@dnn22:~/Donkey3DGS/input_video.mov

mkdir -p ~/Donkey3DGS/frames
ffmpeg -y -i ~/Donkey3DGS/input_video.mov -vf fps=5 -q:v 2 ~/Donkey3DGS/frames/row1_%05d.jpg
```

### 2. Run COLMAP photogrammetry (headless)
```bash
export QT_QPA_PLATFORM=offscreen
mkdir -p ~/Donkey3DGS/colmap/{sparse,dense}
rm -f ~/Donkey3DGS/colmap/database.db

# Feature extraction
colmap feature_extractor \
  --database_path  ~/Donkey3DGS/colmap/database.db \
  --image_path     ~/Donkey3DGS/frames \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --ImageReader.default_focal_length_factor 1.2 \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.max_image_size 1600

# Matching
colmap sequential_matcher \
  --database_path ~/Donkey3DGS/colmap/database.db \
  --SiftMatching.use_gpu 0 \
  --SequentialMatching.overlap 5 \
  --SiftMatching.guided_matching 1

# Quick counts (optional)
sqlite3 ~/Donkey3DGS/colmap/database.db "SELECT COUNT(*) FROM images;"
sqlite3 ~/Donkey3DGS/colmap/database.db "SELECT COUNT(*) FROM matches;"

# Sparse reconstruction
colmap mapper \
  --database_path ~/Donkey3DGS/colmap/database.db \
  --image_path    ~/Donkey3DGS/frames \
  --output_path   ~/Donkey3DGS/colmap/sparse

# Undistort for dense
colmap image_undistorter \
  --image_path ~/Donkey3DGS/frames \
  --input_path ~/Donkey3DGS/colmap/sparse/0 \
  --output_path ~/Donkey3DGS/colmap/dense \
  --output_type COLMAP
```

### 3. Ensure layout for 3DGS (need `dense/sparse/0/`)
```bash
# If you only have dense/sparse/, convert to dense/sparse/0/
mkdir -p ~/Donkey3DGS/colmap/dense/sparse/0
colmap model_converter \
  --input_path  ~/Donkey3DGS/colmap/dense/sparse \
  --output_path ~/Donkey3DGS/colmap/dense/sparse/0 \
  --output_type BIN
```

---

## 🚀 Training 3DGS

```bash
cd ~/Donkey3DGS/gaussian-splatting
conda activate gs-env

# (re-export; harmless if already set)
export TORCH_CUDA_ARCH_LIST="8.6"
export TCNN_CUDA_ARCHITECTURES=86

python train.py \
  -s ~/Donkey3DGS/colmap/dense \
  -m ~/Donkey3DGS/output/ugv_side \
  --resolution 1 \
  --save_iterations 3000 7000 12000 20000 30000
```
If VRAM is tight, try  `--resolution 2`.

- Checkpoints and `.ply` models will appear under:
  ```
  ~/Donkey3DGS/output/ugv_side/point_cloud/
  ```

---

## 🎬 Rendering
Render a checkpoint (e.g., 7000)
```bash
python render.py \
  -m ~/Donkey3DGS/output/ugv_side \
  -s ~/Donkey3DGS/colmap/dense \
  --iteration 7000
# Optional flags: --white_background  --skip_train  --skip_test

```

Options:
- `--white_background` → renders on white background
- `--skip_train` → render only test views
- `--skip_test` → render only train views

---

## 📊 Monitoring

(Optional) Install TensorBoard:
```bash
pip install tensorboard   # No need to install every time
tensorboard --logdir ~/Donkey3DGS/output/ugv_side --port 6006
```

---

## ✅ Tips
- Use `--resolution 2` if VRAM runs out (half res).
- Training quality improves steadily; check renders around **7k–12k iters** and again near **30k iters**.
- If `AssertionError: Could not recognize scene type!` → ensure `dense/sparse/0/` exists and has `cameras.bin`, `images.bin`, `points3D.bin`.

---

## 📂 Final Project Structure

```
Donkey3DGS/
 ├── frames/               # Extracted images
 ├── colmap/
 │   ├── database.db
 │   ├── sparse/0/         # Sparse model
 │   └── dense/
 │       ├── images/       # Undistorted frames
 │       ├── sparse/0/     # Copy of model for 3DGS
 │       └── stereo/       # Depth maps
 ├── gaussian-splatting/   # Repo
 └── output/
     └── ugv_side/         # Training + results
```
