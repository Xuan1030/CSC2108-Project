# CSC2515 Project: Deep Learning Analysis and Improvements on DeepSeek-Prover-V1.5-Base

### **Group Members**:
- **Xuan Xiong**
- **Shaohong Chen**

---

## **Project Overview**

In this project, we analyze and improve upon the **DeepSeek-Prover-V1.5-Base** model, a cutting-edge deep learning framework for genomic and biological data. The model is designed to identify patterns in genomic sequences and predict key biological outcomes. Our goal is to extend its functionality, enhance its accuracy, and make it more interpretable and efficient.

---

## **Project Objectives**

1. **Baseline Understanding**:
   - Reproduce results from the original **DeepSeek-Prover-V1.5-Base** model.
   - Evaluate its performance on publicly available datasets, such as ENCODE or GEO.

2. **Improvements**:
   - Enhance model architecture by integrating modern techniques like transformers or attention mechanisms.
   - Add domain-specific preprocessing steps for genomic data.
   - Improve computational efficiency through model pruning, quantization, or distillation.

3. **Explainability**:
   - Implement methods to interpret model predictions, such as saliency maps or SHAP.

4. **Robustness**:
   - Evaluate model generalization across diverse datasets and noise levels.

---

## **Project Methodology**

1. **Data Collection**:
   - Utilize datasets such as **ENCODE**, **GEO**, or others relevant to genomic analysis.
   - Preprocess data to create training, validation, and test splits.

2. **Baseline Model Setup**:
   - Clone the **DeepSeek-Prover-V1.5-Base** repository from Hugging Face.
   - Reproduce the original model's results to establish a baseline.

3. **Enhancements**:
   - Modify the model architecture to incorporate:
     - Attention mechanisms.
     - Hybrid models (e.g., transformers with convolutional layers).
   - Add preprocessing steps for noise reduction and feature engineering.

4. **Training**:
   - Train the model using optimized hyperparameters (e.g., learning rate, batch size).
   - Implement early stopping and learning rate scheduling.

5. **Evaluation**:
   - Use metrics such as ROC-AUC, F1-Score, and accuracy to evaluate performance.
   - Test the model's interpretability and robustness.

6. **Documentation**:
   - Provide visualizations of model performance, interpretability outputs, and training progress.

---

## **Project Results**

### Baseline Model:
- Accuracy: `XX%`
- F1-Score: `XX%`

### Improved Model:
- Accuracy: `XX%`
- F1-Score: `XX%`
- Computational Efficiency: `XX% reduction in inference time`

---

## **Key Challenges**

1. **Data Preprocessing**:
   - Handling noisy or missing genomic data.
   - Balancing datasets for fair evaluation.

2. **Model Training**:
   - Managing computational resource constraints.
   - Tuning hyperparameters for optimal performance.

3. **Explainability**:
   - Extracting biologically meaningful insights from the model predictions.

---

## **Future Work**

- Extend the model to multi-modal datasets, incorporating clinical or phenotypic data.
- Fine-tune on task-specific datasets for broader applicability.
- Deploy the model as a web-based tool for real-world use.

---

## **How to Run the Code**

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
