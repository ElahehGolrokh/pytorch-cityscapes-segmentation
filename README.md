# 🚀 Cityscapes Semantic Segmentation (PyTorch)

PyTorch implementation of semantic segmentation for urban scenes using a Cityscapes-style dataset.

**Dataset:** [Kaggle - Cityscapes Depth & Segmentation](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data)

---

## 🔑 Key Features

- **Class remapping:** 19 → 10 classes to address severe class imbalance
- **Flexible inference:** Single images, directories, or videos
- **Memory monitoring:** Automatic stopping when RAM threshold exceeded
- **Comprehensive evaluation:** Per-class and aggregate metrics (IoU, F1, precision, recall)
- **Modular design:** Separate modules for training, evaluation, and inference

---

## 💻 Code and Resources Used
- **Python Version**: 3.10.14 
- **Packages**: PyTorch, segmentation-models-pytorch, albumentations, opencv-python, psutil (for memory monitoring)

---

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

---

## 📌 Quick Start

### Training
```bash
python train.py --config config.yaml
```

### Evaluation
```bash
python evaluate.py --config config.yaml -s --report training.epochs
```

**Arguments for `evaluate.py`:**

| Short | Long        | Description | Default |
|-------|------------|-------------|--------|
| `-s`  | `--save_flag`  | specifies whether to save evaluation metrics| False
| `-on`  | `--output_name`  | Name of the output file for evaluation metrics| evaluation_metrics.txt |
| -  | `--report`    | List of config keys to add to the saved report. Example: training.batch_size training.lr| None |

### Inference

**Single image:**
```bash
python inference.py --config config.yaml --image-path path/to/image.png
```

**Image directory:**
```bash
python inference.py --config config.yaml --image-path path/to/images/
```

**Video with memory monitoring:**
```bash
python inference.py --config config.yaml --video-path video.mp4 --memory-threshold 80
```

**Arguments for `inference.py`:**

| Short | Long        | Description | Default |
|-------|------------|-------------|--------|
| `-ip`  | `--image-path`  | Path to the input image file or the directory containing images| None
| `-vp`  | `--video-path`  | Path to the input video file| None |
| `-n`  | `--number_of_visualizations`    | Number of visualizations to generate for image inference | None |
| `-mt`  | `--memory-threshold`    | Memory usage threshold for processing video | 80.0 |


---

## 📦 Project Structure
```
├── src/
│   ├── data_loader.py       # Dataset and data loading
│   ├── preprocessing.py     # Image preprocessing
│   ├── model_building.py    # Model architecture
│   ├── training.py          # Training loop
│   ├── evaluating.py        # Metrics calculation
│   ├── prediction.py        # Inference engine
│   ├── visualization.py     # Result visualization
│   ├── video_utils.py       # Video I/O
│   ├── memory_utils.py      # Memory monitoring
│   └── utils.py             # Helper functions
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── inference.py             # Inference script
└── config.yaml              # Configuration file
```

---

## 🔍 Class Mapping Rationale

The original 19-class dataset suffered from:
- Severe class imbalance
- Rare/missing classes in many scenes
- Poor model convergence

**Solution:** Merged similar/rare classes into 10 categories which are much more balanced:
- `sidewalk` → road surface
- `wall`, `fence`, `building` → structures
- `traffic_light`, `traffic_sign` → traffic objects
- `truck`, `bus`, `train` → large vehicles
- Rare classes (`bicycle`) → background

Full mapping in `config.yaml`.

---

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Dataset paths and class mapping
- Model architecture (backbone of the UNet model)
- Training hyperparameters (lr, batch size, epochs)
- Inference options (batch size, memory threshold)

---

## ⚖️ License

This project is open-source and distributed under the **MIT License**.  
Feel free to use, modify, and share it for research or personal projects.

---

## 🙌 Acknowledgements

- [Cityscapes Depth & Segmentation Dataset](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- PyTorch community

---

**Author:** Elaheh Golrokh  
📧 For questions or collaboration: [GitHub Profile](https://github.com/elahehgolrokh) <br>
🌐 To see portfolio & other projects [click here](https://github.com/elahehgolrokh)
