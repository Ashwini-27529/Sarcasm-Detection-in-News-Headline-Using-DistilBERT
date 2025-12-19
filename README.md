# Sarcasm-Detection-in-News-Headline-Using-DistilBERT
MCA Final year Research paper based project - Sarcasm Detection in news headline using DistilBERT a lightweight transformer based approach

## Project Overview
This research addresses the challenge of identifying sarcasm in brief, context-sparse news headlines. While social media platforms offer cues like hashtags and emojis, news headlines rely purely on subtle linguistic incongruity. This project employs **DistilBERT**, a lightweight transformer model, to achieve a balance between high detection accuracy and computational efficiency suitable for real-time deployment.

## Key Features
* **Architectural Efficiency:** Uses DistilBERT, which is 40% smaller and 60% faster than BERT-base while retaining 95-97% of its performance.
* **Precision-Oriented:** The model follows a conservative classification strategy, prioritizing high precision (reliability) to minimize false positives in media monitoring.
* **Interpretability:** Employs visual analytics such as confusion matrices and ROC curves to demystify model behavior.

## Dataset
The dataset used for this project is the **News Headlines Dataset for Sarcasm Detection** from Kaggle. 

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
* **Author:** Rishabh Misra
* **Description:** This dataset contains high-quality headlines from *The Onion* and *HuffPost* for the task of sarcasm detection.
* This project utilizes a curated subset of the **Sarcasm Headlines Dataset**.
* **Total Samples Used:** 3,600 headlines.
* **Preprocessing:** Lowercasing and WordPiece subword tokenization with a fixed sequence length of 64 tokens

## Model Performance
The fine-tuned model achieved the following results on a held-out test set of 600 samples:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 88.0% |
| **Precision (Sarcastic)** | 0.92 |
| **Recall (Sarcastic)** | 0.80 |
| **F1-Score** | 0.86 |
| **ROC-AUC** | 0.96 |

## Visualizations
### 1. Confusion Matrix
The confusion matrix highlights the model's conservative nature, showing that false negatives (missed sarcasm) are more common than false positives.
<img width="638" height="583" alt="Picture1" src="https://github.com/user-attachments/assets/0784aadc-4982-4b4e-bf6f-9658f4a859c3" />

![Confusion Matrix](./path_to_your_image/confusion_![Uploading Picture1.png…]()
matrix.png)

### 2. ROC and Precision-Recall Curves
The high AUC of 0.96 confirms excellent class separability.
![ROC Curve](./path_to_your_image/roc_curve.png)
![Precision-Recall Curve](./path_to_your_image/pr_curve.png)

### 3. Training Dynamics
Curves demonstrating convergence without overfitting.
![Training Curves](./path_to_your_image/training_plot.png)

### Detailed Classification Report
The heatmap below visualizes the precision, recall, and F1-score for both classes. [cite_start]The model achieves an overall accuracy of **88%**, with a particularly high precision of **0.92** for the sarcastic class[cite: 1164, 1165].

![Classification Report Heatmap](./images/classification_heatmap.png)
