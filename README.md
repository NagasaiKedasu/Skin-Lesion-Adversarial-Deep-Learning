# HAM10000 Skin Lesion Classification
## 1. Project Overview
Briefly explain the problem and objective.
This project develops a deep learning-based system for classifying
dermoscopic skin lesion images from the HAM10000 dataset into seven
diagnostic classes.
The project investigates transfer learning, class imbalance handling,
adversarial robustness, explainable AI, ensemble learning, and
multimodal learning.
## 2. Objectives
- Classify skin lesions into 7 diagnostic categories
- Investigate class imbalance
- Compare different CNN and transformer architectures
- Evaluate ensemble approaches
- Investigate explainability using Grad-CAM, Grad-CAM++, and Integrated Gradients
- Explore metadata-fused multimodal classification
- Provide an interactive Streamlit application
## 3. Dataset
Dataset: HAM10000
Describe:
- Number of images
- Seven classes
- Image characteristics
- Metadata used
- Train/validation/test split
- How class imbalance was handled
IMPORTANT:
The dataset is not included in this GitHub repository because of its size.
Explain how the dataset should be obtained and where it should be placed.
Example:
dataset/
├── images/
└── metadata.csv
## 4. Methodology
### Exploratory Data Analysis
Explain the analysis performed.
### Data Preprocessing
Explain:
- resizing
- normalization
- augmentation
- train/validation/test splitting
### Class Imbalance
Explain the technique used:
- weighted loss
- sampling
- focal loss
- augmentation
etc.
### Transfer Learning
Models investigated:
- EfficientNet-B4
- ViT-B/16
- ResNet-50 + CBAM
- DenseNet-169
Briefly explain why these architectures were selected.
### Ensemble Model
Explain how the models were combined using soft voting
and why ensemble learning was used.
### Multimodal Model
Explain how image features and patient metadata were combined.
## 5. Explainable AI
The project uses:
- Grad-CAM
- Grad-CAM++
- Integrated Gradients
Explain that these methods help visualize which image regions
contributed to model predictions.
## 6. Results
### Model Performance
| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| EfficientNet-B4 | ... | ... | ... | ... |
| ViT-B/16 | ... | ... | ... | ... |
| ResNet-50 + CBAM | ... | ... | ... | ... |
| DenseNet-169 | ... | ... | ... | ... |
| Ensemble | ... | ... | ... | ... |
| Multimodal | ... | ... | ... | ... |
**Use your actual values from your experiments here. Don't put estimates.**
Then briefly explain which model performed best and why.
## 7. Project Structure
```text
Skin_Lesion/
├── app.py
├── app_common.py
├── requirements.txt
├── README.md
├── PROJECT_QA_DOCUMENT.md
├── configs/
├── notebooks/
├── results/
├── src/
└── views/

Explain what each folder contains.

8. Streamlit Application

Explain the available pages/features:

* Prediction
* Explainability
* Model Performance
* Dataset Explorer
* Project Q&A

9. Installation

pip install -r requirements.txt

10. Running the Application

streamlit run app.py

Explain any required dataset/model files.

11. Reproducing the Experiments

Give the recommended order:

1. Run EDA notebook
2. Prepare dataset
3. Train models
4. Evaluate models
5. Generate explainability results
6. Run ensemble/multimodal experiments
7. Launch Streamlit application

12. Limitations

Mention things such as:

* Dataset imbalance
* Dataset size/diversity
* Computational requirements
* Dependence on pretrained models
* Results should not be interpreted as clinical diagnosis

13. Ethical Considerations

Explain that the system is an academic/research prototype
and is not intended to replace professional medical diagnosis.

14. Author / Academic Project

Student: Naga Sai Kedasu
Course: Masters Data Science
Institution:SETU Carlow
Academic Year: 2025 to 2026

