# Hybrid Multi-Modal Skin Cancer Classification using CNN, Vision Transformer and Explainable AI

This repository hosts a publication-quality deep learning system designed for the automatic diagnosis of skin cancer. The system integrates dual-backbone image feature extractors (Convolutional Neural Networks + Vision Transformers) and patient metadata (age, sex, localization) using multi-modal feature fusion. To build clinical trust, the predictions are backed by an Explainable AI (XAI) engine generating visual attention/gradient heatmaps (Grad-CAM & self-attention maps) and clinician-friendly reports, wrapped in a Streamlit web application.

---

## Workspace Directory Architecture

Below is the layout of the project directories with their design responsibilities:

```
SkinCancerAI/
│
├── configs/             # Configuration files (YAML/JSON) for hyperparameters and paths
├── dataset/             # Data loading scripts, dataset objects, and raw image symlinks
├── metadata/            # Patient clinical demographics (HAM10000_metadata.tab)
│
├── models/              # Neural network definitions and modular sub-networks
│   ├── cnn/             # CNN feature extraction backbones (e.g., EfficientNet, ResNet)
│   ├── transformer/     # Vision Transformer (ViT) feature extraction backbones
│   ├── fusion/          # Fusion modules combining CNN, Transformer, and Metadata vectors
│   └── multimodal/      # End-to-end multi-modal classifier linking all components
│
├── explainability/      # Explainable AI (XAI) scripts (Grad-CAM, ViT Attention Rollout)
├── training/            # Training loop, loss functions, optimization, and checkpointers
├── evaluation/          # Testing routines, metric calculations (ROC-AUC, Confusion Matrix)
├── reports/             # Automatic clinical PDF report generator and documentation
├── streamlit_app/       # Frontend clinician-friendly dashboard interface
├── utils/               # Seeding, logging configuration, and file helper functions
│
├── checkpoints/         # [Ignored in Git] Locally saved model weights (.pth files)
├── logs/                # [Ignored in Git] TensorBoard logs and plain-text logging files
├── outputs/             # [Ignored in Git] Generated classification plots, figures, and CSV metrics
│
├── tests/               # Unit and integration tests for sanity-checking codebase
├── docs/                # Research papers, base papers, bibliography, and manuscript figures
│
├── requirements.txt     # Complete reproducible python packages manifest
├── .gitignore           # File exclusion patterns to keep version control clean
├── README.md            # Comprehensive project overview and user documentation
└── main.py              # Main CLI entry point to orchestrate training/evaluation
```

---

## Directory Responsibilities

1.  **`configs/`**: centralizes all tunable model variables (epochs, batch sizes, learning rates, architectures, ablation options). This ensures no hyperparameters or file paths are ever hardcoded in the source modules.
2.  **`dataset/`**: manages the PyTorch `Dataset` and `DataLoader` classes. Handles image pre-processing, data augmentation (rotations, color jitter), missing data imputation, and lesion-level split stratifications to prevent data leakage.
3.  **`metadata/`**: houses the raw structured tabular files mapping image IDs to diagnostic targets and demographic attributes.
4.  **`models/`**: acts as the model suite. It is divided into:
    *   `cnn/`: for extracting localized feature patterns (e.g., color, border, texture).
    *   `transformer/`: for capturing global spatial context and long-range patches relationships.
    *   `fusion/`: for blending intermediate latent vectors of multiple modalities.
    *   `multimodal/`: for compiling the complete unified neural network architecture.
5.  **`explainability/`**: calculates gradient and attention heatmaps (via Grad-CAM and ViT Attention maps) to explain which skin lesion features drove the classifier's predictions.
6.  **`training/`**: contains training engine components (mixed-precision loops, loss modules, learning rate schedules, early stopping checks, and checkpointing).
7.  **`evaluation/`**: runs testing routines and computes academic metrics like macro F1-score, sensitivity, specificity, and ROC-AUC curves.
8.  **`reports/`**: formats diagnostic predictions, confidence intervals, and heatmaps into polished report files for medical records.
9.  **`streamlit_app/`**: contains the UI code that acts as a clinic-ready frontend, displaying the heatmaps and reports dynamically.
10. **`utils/`**: holds auxiliary code (e.g., seeding, path validation, printing tables).
11. **`checkpoints/`**, **`logs/`**, and **`outputs/`**: local directories ignored by git, storing temporary model states, training stats, and generated figures.
12. **`tests/`**: stores modular test scripts checking data loader dimensions and forward/backward passes.
13. **`docs/`**: houses PDFs of base papers and drafts of the research manuscript.

---

## Multi-Stage Validation & Calibrated Inference Pipeline

To enforce high clinical trust, the system features a multi-stage validation and calibration inference pipeline before executing disease predictions:

1. **Phase 1: Image Validation (`ImageValidator`):** Verifies image readability, dimensions constraints, solid-color blocks, blur levels (via grayscale Laplacian variance), and filters out cartoons, screenshots, and text documents using edge-alignment (orthogonal Sobel gradients) and color-variance heuristics.
2. **Phase 2: Skin Detection (`SkinDetector`):** Assesses whether human skin tones are present in the image using YCrCb color space bounds.
3. **Phase 3: Lesion Detection (`LesionDetector`):** Verifies if a high-contrast skin lesion contour exists inside the skin mask boundaries using local contrast contours.
4. **Phase 6: Probability Calibration:** Applies Temperature Scaling ($T=1.2$) to calibrate network output probabilities.
5. **Phases 4 & 5: Thresholding & Unknown Class:** If the maximum calibrated probability is below `confidence_threshold` (e.g. 60%), the final classification is overridden to `"unknown"` (Unknown Category) with a warning recommendation, preventing forced disease categorization.
6. **Phase 8: Logging:** Directs telemetry streams across separate log files under `logs/`: `validation.log`, `inference.log`, and `prediction.log`.

