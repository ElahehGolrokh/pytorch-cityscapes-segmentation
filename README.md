# Cityscapes Semantic Segmentation (PyTorch)

This project implements a complete **semantic segmentation pipeline in PyTorch** using a Cityscapes-style dataset from Kaggle:

**Dataset:**  
https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data

The project includes:

- Data preprocessing  
- Class remapping from **19 original classes** to **10 merged classes**
- Model training  
- Evaluation  
- Inference and visualization  

---

## 🚀 Project Overview

The original dataset provides **19 semantic classes**.  
However, training a segmentation model on all 19 classes produced **poor performance**, mainly due to:

- Severe class imbalance  
- Very small or rare classes  
- Missing classes in many scenes  
- Inconsistent annotation density

To improve results, I analyzed the **pixel distribution of all masks** and merged several rare or similar classes, producing a **10-class mapping**.  
This significantly improved stability and model accuracy.

---

## 🗺️ Class Mapping (19 → 10 Classes)

Some examples:

- `sidewalk` merged into road-like category  
- `wall`, `fence`, `building` merged into a single “structure” category  
- `traffic_light` + `traffic_sign` merged into “traffic_object”  
- All large vehicles (`truck`, `bus`, `train`) merged into a single “large vehicle” class  
- `bicycle` mapped to background due to 0 frequency

The full mapping dictionary is provided in `config.yaml`.

---

## ✨ Features

### **Data Preprocessing**
- Loading images and masks  
- Custom class remapping  
- Statistical normalization  
- Data augmentation pipeline  
- Dataset-wide class distribution analysis

### **Model Training**
- PyTorch-based training loop  
- Configurable architecture (UNet with arbitrary backbone)  
- Learning rate scheduling  
- Checkpointing and logging

### **Evaluation**
- Pixel accuracy  
- Class IoU  
- Mean IoU (mIoU)  

### **Inference**
- Single-image prediction  
- Batch prediction  
- Overlay visualization  
- Color-encoded segmentation masks  

---

## 📊 Results Summary

**10-Class Model Significantly improved mIoU**  
- More stable training  
- Better generalization  
- Cleaner segmentation outputs  

Results and visual examples are in the `results/` directory.

---