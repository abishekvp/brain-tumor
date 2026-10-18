# Qubit Brain — Brain Tumor Classification & Segmentation System

An end-to-end **Deep Learning & Quantum Machine Learning** web application built with **Django**, **PyTorch**, **PennyLane**, **TensorFlow/Keras**, and **OpenCV** for automated **Brain Tumor MRI Classification**, **Pixel-Level U-Net Tumor Segmentation**, **Tumor Region Highlighting**, and **Multilingual Medical AI Assistance**.

---

## Table of Contents
1. [System Architecture & Pipeline](#system-architecture--pipeline)
2. [Project Directory Structure](#project-directory-structure)
3. [Issues Identified & Fixed](#issues-identified--fixed)
4. [How to Run the Django Web Application](#how-to-run-the-django-web-application)
5. [How to Build & Train the Models](#how-to-build--train-the-models)
   - [1. Building the Hybrid Quantum-Classical ViT Classifier](#1-building-the-hybrid-quantum-classical-vit-classifier)
   - [2. Building the U-Net Segmentation Models](#2-building-the-u-net-segmentation-models)
   - [3. Deploying Newly Trained Weights to Django](#3-loading-newly-trained-weights-in-django)

---

## System Architecture & Pipeline

When a user uploads a Brain MRI scan in the web portal (`/model/`), the system executes a two-stage diagnostic pipeline:

### Stage 1: 4-Class Brain Tumor Classification (`HybridQuantumViT`)
- **Model**: `HybridQuantumViT` ([`django/App/hybrid_quantum_vit.py`](./django/App/hybrid_quantum_vit.py))
- **Weights**: [`models/classification/best_model.pth`](./models/classification/best_model.pth) (Trained for 66 epochs on GPU; **Validation Macro-F1: 89.41%**)
- **Classes** (alphabetical `ImageFolder` order):
  1. `Glioma`
  2. `Meningioma`
  3. `No Tumor`
  4. `Pituitary`
- **Architecture**:
  1. **Patch Embedding**: Splits a `224 × 224 × 3` MRI scan into `196` non-overlapping patches of `16 × 16` pixels and projects them to `embed_dim = 128`.
  2. **Classical ViT Backbone**: Prepends a learnable `[CLS]` token + positional embeddings and passes tokens through `6` Transformer Encoder blocks (`4` attention heads, `MLP ratio = 4.0`, `GELU`, `LayerNorm`).
  3. **Quantum Bottleneck Projection**: Projects the 128-D `[CLS]` token to `6` dimensions (`LayerNorm -> Linear(128, 6) -> Tanh`) scaled by $\pi$ for angle encoding.
  4. **Variational Quantum Circuit (PennyLane)**:
     - **6 Qubits** (`lightning.qubit` / `default.qubit`)
     - **Encoding**: `qml.AngleEmbedding` with $R_Y$ rotations
     - **Variational Ansatz**: `3` layers of parameterized single-qubit `qml.Rot(φ, θ, ω)` gates followed by ring-topology controlled-$R_X$ (`qml.CRX`) entangling gates
     - **Measurement**: Pauli-$Z$ expectation values $\langle Z_0 \rangle, \dots, \langle Z_5 \rangle \in [-1, 1]^6$
  5. **Residual Quantum-Classical Fusion & Head**: Projects quantum features back to 128-D (`Linear(6, 128) -> GELU -> Dropout`) and fuses them with the classical `[CLS]` feature via residual addition (`fused = cls_feat + q_feat`), followed by `LayerNorm -> Linear(128, 4)` and `Softmax` to obtain class probabilities and confidence percentage.

### Stage 2: Pixel-Level Tumor Segmentation & Region Highlighting (`U-Net`)
- **Models**: Category-specific 2D U-Net models stored in `models/segmentation/`:
  - `Glioma` $\rightarrow$ [`models/segmentation/Glioma.h5`](./models/segmentation/Glioma.h5)
  - `Meningioma` $\rightarrow$ [`models/segmentation/Meningioma.h5`](./models/segmentation/Meningioma.h5)
  - `Pituitary` $\rightarrow$ [`models/segmentation/pituitory.h5`](./models/segmentation/pituitory.h5)
  - `No Tumor` $\rightarrow$ [`models/segmentation/No Tumor.h5`](./models/segmentation/No%20Tumor.h5)
- **Architecture**:
  - **Input**: `128 × 128 × 3` normalized RGB MRI image
  - **Encoder**: 4 downsampling stages (`64 -> 128 -> 256 -> 512` filters) with `Conv2D(3×3) + BatchNormalization + ReLU` and `MaxPool2D(2×2)`
  - **Bridge**: `1024` filters
  - **Decoder**: 4 upsampling stages (`512 -> 256 -> 128 -> 64` filters) using `Conv2DTranspose(2×2, stride=2)` concatenated with encoder skip connections
  - **Output**: `Conv2D(1, 1×1, activation='sigmoid')` producing a pixel-wise tumor probability map
- **Highlighting & Post-Processing (`create_highlighted_tumor_images` in [`django/App/views.py`](./django/App/views.py))**:
  - Smoothly upsamples the `128 × 128` U-Net probability map to `320 × 320` via bicubic interpolation.
  - Applies primary thresholding (`p > 0.5`) with an adaptive threshold fallback (`max(0.12, max_prob * 0.60)`) and morphological smoothing (`MORPH_CLOSE` + `MORPH_OPEN`) so lower-contrast tumor activations are never lost.
  - Generates **three visual outputs** displayed side-by-side on the result page:
    1. **Original Brain MRI**
    2. **U-Net Binary Segmentation Mask**
    3. **Highlighted Tumor Region Overlay** — blends a class-coded semi-transparent color fill over the tumor pixels, traces the exact tumor boundary contour in bright cyan/green, and draws a yellow bounding box with the predicted class label and confidence score.

---

## Project Directory Structure

```text
brain-tumor/
├── README.md                                      # Project documentation (this file)
├── models/                                        # Centralized directory for all built/trained models
│   ├── classification/
│   │   ├── best_model.pth                         # Trained HybridQuantumViT checkpoint (Epoch 66, Val F1: 0.894)
│   │   └── LENET.h5                               # Baseline LeNet classifier weights
│   ├── segmentation/
│   │   ├── Glioma.h5                              # Trained U-Net weights for Glioma segmentation
│   │   ├── Meningioma.h5                          # Trained U-Net weights for Meningioma segmentation
│   │   ├── pituitory.h5                           # Trained U-Net weights for Pituitary segmentation
│   │   └── No Tumor.h5                            # Trained U-Net weights for No Tumor
│   └── chatbot/
│       ├── chatbot_model.h5                       # Trained Keras intent classification model
│       ├── words.pkl                              # Chatbot tokenized vocabulary
│       └── classes.pkl                            # Chatbot intent labels
├── django/                                        # Django Web Application root
│   ├── manage.py                                  # Django CLI entrypoint
│   ├── requirements.txt                           # Python dependencies
│   ├── db.sqlite3                                 # SQLite database
│   ├── Project/                                   # Django project configuration
│   │   ├── settings.py                            # App settings, MODELS_DIR, media/static configuration
│   │   ├── urls.py                                # Root URL routing
│   │   └── wsgi.py / asgi.py
│   ├── App/                                       # Core Brain Tumor Classification & Segmentation app
│   │   ├── hybrid_quantum_vit.py                  # PyTorch + PennyLane HybridQuantumViT model definition
│   │   ├── views.py                               # Inference, tumor highlighting, auth & database views
│   │   ├── models.py                              # UserImageModel & Profile database models
│   │   ├── forms.py                               # Upload & authentication forms
│   │   ├── urls.py                                # App URL routes
│   │   └── templates/                             # HTML templates (home.html, 9_Deploy.html, output.html, Database.html)
│   └── Chatbot/                                   # Multilingual Medical AI Assistant app
│       ├── intents.json                           # Medical Q&A intents dataset
│       ├── processor.py                           # NLP tokenization, intent inference & TTS helper
│       ├── views.py                               # Multilingual translation & chat API endpoint
│       └── templates/chatbot/widget.html          # Floating voice + text chat widget
└── quantum vit model/                             # Model Training Notebooks, Scripts & Datasets
    ├── hybrid_quantum_vit_explained.pdf           # Detailed architectural explanation of HybridQuantumViT
    ├── train_hybrid_quantum_vit_colab.ipynb       # Google Colab GPU training notebook for HybridQuantumViT
    ├── Classifications/
    │   ├── train_hybrid_quantum_vit.py            # Standalone CLI training script for HybridQuantumViT
    │   ├── train_hybrid_quantum_vit.ipynb         # Local Jupyter notebook for HybridQuantumViT
    │   └── DATASET/                               # Classification dataset (TRAIN/ and VAL/ split by 4 classes)
    └── Segmentations/
        ├── train_unet_segmentation.py             # Standalone CLI training script for U-Net segmentation
        ├── Glioma/                                # Glioma U-Net notebook (Glioma.ipynb) & DATASET/
        ├── Meningioma/                            # Meningioma U-Net notebook (Meningioma.ipynb) & DATASET/
        ├── Pituitary/                             # Pituitary U-Net notebook (Pituitary.ipynb) & DATASET/
        └── No Tumor/                              # No Tumor U-Net notebook (No_Tumor.ipynb) & DATASET/
```

---

## Issues Identified & Fixed

During the project audit and live evaluation, the following issues were identified and resolved:

1. **Centralized Model Storage (`models/`)**:
   - Moved all built classification (`best_model.pth`, `LENET.h5`), segmentation (`Glioma.h5`, `Meningioma.h5`, `pituitory.h5`, `No Tumor.h5`), and chatbot (`chatbot_model.h5`, `words.pkl`, `classes.pkl`) models out of the source code directories into a dedicated [`models/`](./models) folder at the project root, configured via `MODELS_DIR` in [`django/Project/settings.py`](./django/Project/settings.py), and removed duplicate `.h5` copies (~1.5 GB saved).
2. **Tumor Region Was Not Highlighted on the Brain MRI (`django/App/views.py` & `output.html`)**:
   - *Issue*: Previously, `views.py` only converted the raw `128 × 128` binary `0/1` array (`y_pred > 0.5`) into a grayscale black-and-white image (`to_image(y_pred)`), displaying a white blob on a solid black square without overlaying it on the brain MRI scan. Furthermore, if a tumor scan had a peak U-Net probability just below `0.5` (e.g., `0.481` on validation Glioma scans), the rigid `> 0.5` threshold produced an all-black image with zero highlighted pixels.
   - *Fix*: Implemented `create_highlighted_tumor_images()` in [`django/App/views.py`](./django/App/views.py), which upsamples the probability map to `320 × 320`, applies adaptive thresholding + morphological smoothing, overlays a class-specific semi-transparent color highlight on the original MRI, draws the exact tumor contour and bounding box with confidence label, and displays all 3 views (**Original Brain MRI**, **U-Net Segmentation Mask**, and **Highlighted Tumor Region**) in [`django/App/templates/app/output.html`](./django/App/templates/app/output.html).
3. **Hardcoded `98.7%` Confidence Score (`output.html` & `views.py`)**:
   - *Issue*: `output.html` had a hardcoded `Confidence: 98.7%` string instead of computing the actual model probability.
   - *Fix*: Computed real softmax probabilities (`torch.softmax(logits, dim=1)`) and segmented tumor area percentage (`tumor_area_pct`) in `views.py` and rendered them dynamically in `output.html`.
4. **Missing `MEDIA_URL` and `MEDIA_ROOT` Causing `ImproperlyConfigured` Crash (`django/Project/settings.py`)**:
   - *Issue*: `Project/urls.py` called `static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)`, but `MEDIA_URL` was not defined in `settings.py` (defaulting to `""`), which causes Django to raise `ImproperlyConfigured: Empty static prefix not permitted`.
   - *Fix*: Added `MEDIA_URL = "/media/"`, `MEDIA_ROOT = BASE_DIR`, and `ALLOWED_HOSTS` in [`django/Project/settings.py`](./django/Project/settings.py).
5. **Missing `social_django` Dependency in `django/Project/urls.py`**:
   - *Issue*: `Project/urls.py` included `social_django.urls` even though `social-auth-app-django` is not used or installed, crashing server startup with `ModuleNotFoundError`.
   - *Fix*: Removed the unused `oauth/` route from [`django/Project/urls.py`](./django/Project/urls.py).
6. **PennyLane + Autoray Incompatibility (`AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'`)**:
   - *Issue*: `pennylane==0.35.1` fails to import when paired with newer `autoray>=0.11.0`.
   - *Fix*: Pinned `pennylane==0.35.1`, `pennylane-lightning==0.35.1`, and `autoray==0.6.12` in [`django/requirements.txt`](./django/requirements.txt) and removed unused conflicting packages (`gradio`, `openai`, `groq`, `firebase-admin`, `google-genai`) that caused `pip` dependency resolution backtracking.
7. **Chatbot Hardcoded `E:/...` Path & URL Shadowing (`django/Chatbot/processor.py`, `App/urls.py`)**:
   - *Issue*: `Chatbot/processor.py` had a hardcoded `save_path="E:/2025 - 2026/..."` and unguarded `pygame.mixer` imports, and `App/urls.py` shadowed `/chatbot/` with a view that ignored language selection.
   - *Fix*: Updated [`django/Chatbot/processor.py`](./django/Chatbot/processor.py) to load models from `models/chatbot/`, use portable paths, automatic NLTK resource checks, and resilient translation (`deep_translator` with `googletrans` fallback) in [`django/Chatbot/views.py`](./django/Chatbot/views.py).
8. **Training Script / Notebook Bugs (`quantum vit model/`)**:
   - *Issue*: In the local classification notebook, `CLASS_NAMES` had `"Pituitary"` and `"No Tumor"` swapped relative to `ImageFolder` alphabetical order; in the segmentation notebooks, `glob.glob(ori_images + "**/*.jpg")` was missing a path separator `/` before `**` and did not sort image/mask file lists before splitting.
   - *Fix*: Created clean, standalone CLI training scripts [`quantum vit model/Classifications/train_hybrid_quantum_vit.py`](./quantum%20vit%20model/Classifications/train_hybrid_quantum_vit.py) and [`quantum vit model/Segmentations/train_unet_segmentation.py`](./quantum%20vit%20model/Segmentations/train_unet_segmentation.py) with these bugs fixed and configured to save directly to `models/classification/` and `models/segmentation/`.

---

## How to Run the Django Web Application

### Prerequisites
- **Python**: `3.10` or `3.11` (64-bit recommended)
- **OS**: Windows, Linux, or macOS

### Step 1: Navigate to the Django Directory & Activate the Virtual Environment
Open PowerShell or Command Prompt:

```powershell
cd "c:\Users\AbishekV\Documents\brain-tumor\django"

# Create a virtual environment if not already created:
python -m venv env

# Activate the virtual environment (Windows PowerShell):
.\env\Scripts\Activate.ps1
```

### Step 2: Install Required Packages
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run Database Migrations & System Check
```powershell
python manage.py migrate
python manage.py check
```

### Step 4: Start the Development Server
```powershell
python manage.py runserver
```

### Step 5: Access the Application in Your Browser
1. Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**.
2. Click **Register** (`/register/`) to create an account or **Login** (`/login/`) with an existing account.
3. Click **Model** (`/model/`) in the navigation bar.
4. Upload any Brain MRI image (e.g., from `quantum vit model/Classifications/DATASET/VAL/{Glioma,Meningioma,No Tumor,Pituitary}/`) and click **Analyze Image**.
5. View the diagnosis result page displaying:
   - **Original Brain MRI**
   - **U-Net Segmentation Mask**
   - **Highlighted Tumor Region** (with color overlay, boundary contour, and bounding box)
   - **Predicted Tumor Class, Confidence Score (%), Highlighted Region Area (%), and Clinical Guidance Table**
6. Use the floating **Brain Tumor AI Assistant** button in the bottom-right corner to ask questions via text or voice in English, Hindi, Tamil, Telugu, Kannada, or Malayalam.

---

## How to Build & Train the Models

The project uses two types of deep learning models trained from the [`quantum vit model/`](./quantum%20vit%20model) directory, which automatically save their built checkpoints into the centralized [`models/`](./models) directory:
1. **Hybrid Quantum-Classical Vision Transformer (`HybridQuantumViT`)** $\rightarrow$ saved to `models/classification/best_model.pth`.
2. **2D U-Net Models** $\rightarrow$ saved to `models/segmentation/{Glioma.h5, Meningioma.h5, pituitory.h5, No Tumor.h5}`.

---

### 1. Building the Hybrid Quantum-Classical ViT Classifier

#### Dataset Structure
Ensure your classification dataset is organized in `ImageFolder` format under `quantum vit model/Classifications/DATASET/`:
```text
quantum vit model/Classifications/DATASET/
├── TRAIN/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── No Tumor/
│   └── Pituitary/
└── VAL/
    ├── Glioma/
    ├── Meningioma/
    ├── No Tumor/
    └── Pituitary/
```

#### Option A: Train via Command Line Script (Recommended)
Run [`train_hybrid_quantum_vit.py`](./quantum%20vit%20model/Classifications/train_hybrid_quantum_vit.py):

```powershell
cd "c:\Users\AbishekV\Documents\brain-tumor\quantum vit model\Classifications"

..\..\django\env\Scripts\python.exe train_hybrid_quantum_vit.py `
    --data-dir ./DATASET `
    --epochs 75 `
    --batch-size 32 `
    --lr 0.0003
```
- The script automatically uses a CUDA GPU (with mixed precision `AMP`) if available, or falls back to CPU.
- By default, it saves the best checkpoint directly to `c:\Users\AbishekV\Documents\brain-tumor\models\classification\best_model.pth`.

#### Option B: Train via Google Colab / Jupyter Notebook
- Open [`quantum vit model/train_hybrid_quantum_vit_colab.ipynb`](./quantum%20vit%20model/train_hybrid_quantum_vit_colab.ipynb) in Google Colab (with a T4/L4 GPU runtime) or [`quantum vit model/Classifications/train_hybrid_quantum_vit.ipynb`](./quantum%20vit%20model/Classifications/train_hybrid_quantum_vit.ipynb) locally.
- Run all cells sequentially to train, plot loss/accuracy/F1 curves, evaluate the confusion matrix, and export `best_model.pth` into `models/classification/best_model.pth`.

---

### 2. Building the U-Net Segmentation Models

#### Dataset Structure
Each tumor category under `quantum vit model/Segmentations/<Category>/DATASET/` contains paired MRI images and ground-truth binary masks:
```text
quantum vit model/Segmentations/<Category>/DATASET/
└── TRAIN/
    ├── original/     # Input MRI images (.jpg / .png)
    └── mask/         # Corresponding ground-truth binary tumor masks (.jpg / .png)
```
*(where `<Category>` is `Glioma`, `Meningioma`, `Pituitary`, or `No Tumor`)*.

#### Option A: Train via Command Line Script (Recommended)
Run [`train_unet_segmentation.py`](./quantum%20vit%20model/Segmentations/train_unet_segmentation.py) for each category (by default it saves directly into `models/segmentation/`):

```powershell
cd "c:\Users\AbishekV\Documents\brain-tumor\quantum vit model\Segmentations"

# 1. Train Glioma U-Net -> saves to ../../models/segmentation/Glioma.h5
..\..\django\env\Scripts\python.exe train_unet_segmentation.py --category Glioma --epochs 50 --batch-size 8

# 2. Train Meningioma U-Net -> saves to ../../models/segmentation/Meningioma.h5
..\..\django\env\Scripts\python.exe train_unet_segmentation.py --category Meningioma --epochs 50 --batch-size 8

# 3. Train Pituitary U-Net -> saves to ../../models/segmentation/pituitory.h5
..\..\django\env\Scripts\python.exe train_unet_segmentation.py --category Pituitary --epochs 50 --batch-size 8

# 4. Train No Tumor U-Net -> saves to "../../models/segmentation/No Tumor.h5"
..\..\django\env\Scripts\python.exe train_unet_segmentation.py --category "No Tumor" --epochs 50 --batch-size 8
```

#### Option B: Train via Jupyter Notebooks
Open and run the notebooks in each folder:
- [`quantum vit model/Segmentations/Glioma/Glioma.ipynb`](./quantum%20vit%20model/Segmentations/Glioma/Glioma.ipynb)
- [`quantum vit model/Segmentations/Meningioma/Meningioma.ipynb`](./quantum%20vit%20model/Segmentations/Meningioma/Meningioma.ipynb)
- [`quantum vit model/Segmentations/Pituitary/Pituitary.ipynb`](./quantum%20vit%20model/Segmentations/Pituitary/Pituitary.ipynb)
- [`quantum vit model/Segmentations/No Tumor/No_Tumor.ipynb`](./quantum%20vit%20model/Segmentations/No%20Tumor/No_Tumor.ipynb)

---

### 3. Loading Newly Trained Weights in Django

Because the training scripts output directly to [`models/classification/`](./models/classification) and [`models/segmentation/`](./models/segmentation), and Django's [`settings.MODELS_DIR`](./django/Project/settings.py) reads directly from [`models/`](./models), no manual file copying is needed—simply restart `python manage.py runserver` to load any newly trained weights.