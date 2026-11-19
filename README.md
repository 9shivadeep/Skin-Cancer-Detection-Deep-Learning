# Skin Cancer Detection Using Deep Learning

An end-to-end automated skin lesion classification web application built with **PyTorch** and **Flask**. The system combines **ResNet-18** and **MobileNetV2** into a soft-voting ensemble model to classify **14 distinct types of skin lesions** from clinical and dermoscopic imagery, providing real-time top-3 probability scores through a lightweight web interface.

---

## 1. Problem Statement & Motivation
Early detection of malignant skin lesions such as **Melanoma** and **Basal Cell Carcinoma** significantly improves patient survival rates. However, visual inspection of skin lesions is challenging because benign conditions often share visual similarities with malignant tumors. This project builds an automated clinical decision-support prototype using transfer learning and ensemble deep learning to classify 14 skin conditions quickly and consistently.

---

## 2. Dataset & Preprocessing Pipeline
* **Dataset Source:** Merged **HAM10000 + MSLDv2.0** dataset (`ahmedxc4/skin-ds`), structured into `train`, `val`, and `test` splits (`3,674` unseen test images).
* **Target Classes (14):** `Actinic keratoses`, `Basal cell carcinoma`, `Benign keratosis-like lesions`, `Chickenpox`, `Cowpox`, `Dermatofibroma`, `HFMD`, `Healthy`, `Measles`, `Melanocytic nevi`, `Melanoma`, `Monkeypox`, `Squamous cell carcinoma`, and `Vascular lesions`.
* **Image Preprocessing:**
  1. Convert input images to standard **3-channel RGB** format.
  2. Resize images uniformly to **`64 × 64` pixels** for fast inference.
  3. Convert pixel values to PyTorch tensors (`[0, 1]`) and normalize using `mean = [0.5, 0.5, 0.5]` and `std = [0.5, 0.5, 0.5]`.

---

## 3. System Architecture & Workflow

```text
[User Uploads Skin Image via Browser (templates/index.html)]
                          │
                          ▼
        [Flask Backend Server (app.py)]
                          │
                          ▼
 [Image Preprocessing: RGB -> Resize 64x64 -> Normalize]
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
   [Pre-trained ResNet-18]   [Pre-trained MobileNetV2]
    (Residual Connections)   (Depthwise Separable Conv)
             │                         │
             ▼                         ▼
      [14-Class Logits]         [14-Class Logits]
             └────────────┬────────────┘
                          ▼
        [Ensemble Averaging: (Logits1 + Logits2) / 2]
                          │
                          ▼
         [Softmax Probability Distribution]
                          │
                          ▼
   [Top-3 Predicted Classes & Confidence Bars Rendered]
```

### Why an Ensemble of ResNet-18 + MobileNetV2?
* **ResNet-18** uses residual skip connections to prevent vanishing gradients and capture deeper spatial textures and lesion borders.
* **MobileNetV2** uses inverted residuals and depthwise separable convolutions, making it computationally lightweight and sensitive to fine-grained local patterns.
* **Logit Averaging Ensemble:** Combining both architectures reduces individual model variance and consistently outperforms either standalone model on unseen validation and test data.

---

## 4. Training Setup & Evaluation Results
* **Training Environment:** Google Colab (NVIDIA T4 GPU / CUDA)
* **Loss Function & Optimizer:** Cross-Entropy Loss with Adam Optimizer (`lr = 0.001`, `batch_size = 64`)
* **Validation & Test Performance:**

| Model Architecture | Best Validation Accuracy | Unseen Test Accuracy (3,674 Images) |
| :--- | :---: | :---: |
| **ResNet-18** (Transfer Learning) | 75.66% | — |
| **MobileNetV2** (Transfer Learning) | 79.13% | — |
| **Ensemble (ResNet-18 + MobileNetV2)** | **81.07%** | **81.00%** (Weighted F1: `0.80`) |

---

## 5. Repository Structure
* `train_model.ipynb` — Complete Google Colab pipeline for dataset loading, preprocessing, training ResNet-18 and MobileNetV2, evaluating on the test set, and exporting model weights (`ensemble_trained.pth`).
* `app.py` — Flask web application server that loads the trained ensemble checkpoint, processes uploaded images, and computes Softmax probabilities.
* `templates/index.html` — Responsive frontend interface for uploading skin lesion images and visualizing top-3 predictions with confidence bars.
* `class_names.json` — Ordered list of the 14 target skin lesion labels.
* `requirements.txt` — Python dependencies (`flask`, `torch`, `torchvision`, `pillow`, `numpy`).

---

## 6. How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd Skin-Cancer-Detection-Deep-Learning
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place your trained weights (`ensemble_trained.pth`) in the root folder and start the Flask server:**
   ```bash
   python app.py
   ```
4. **Open in your browser:** Navigate to `http://127.0.0.1:5000`.

---

## 7. Known Limitations & Future Improvements
* **Domain Shift:** Trained primarily on close-up clinical and dermoscopic lesion images; wide-angle smartphone photos with background lighting or hair artifacts can cause false positives.
* **Class Imbalance:** Rare classes (such as *Dermatofibroma* and *Squamous cell carcinoma*) have fewer training samples than common classes like *Melanocytic nevi*.
* **Planned Upgrades:** Increase input resolution from `64×64` to `224×224`, apply focal loss or class-weighted sampling, and add an out-of-distribution confidence threshold.

> **Disclaimer:** This project is an academic deep learning prototype built for educational and research purposes only and is not intended for clinical medical diagnosis.
