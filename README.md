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


<img width="338" height="383" alt="Picture1" src="https://github.com/user-attachments/assets/0784aadc-4982-4b4e-bf6f-9658f4a859c3" />



### 2. ROC and Precision-Recall Curves
The high AUC of 0.96 confirms excellent class separability.

<img width="338" height="383" alt="Picture3" src="https://github.com/user-attachments/assets/eb0bd125-3216-49e1-870e-6e51ebefbef7" />

<img width="338" height="383" alt="Picture4" src="https://github.com/user-attachments/assets/68ffad01-9689-4ada-8b38-b8e4b5b28748" />



### 3. Training Dynamics
Curves demonstrating convergence without overfitting.

<img width="400" height="426" alt="Picture2" src="https://github.com/user-attachments/assets/f2d145a0-9008-44e6-9910-c360c139cb64" />


### Detailed Classification Report
The heatmap below visualizes the precision, recall, and F1-score for both classes. The model achieves an overall accuracy of **88%**, with a particularly high precision of **0.92** for the sarcastic class.

<img width="338" height="383" alt="Picture5" src="https://github.com/user-attachments/assets/76997b4f-5d0d-4490-ba3f-e9277ccfaf8a" />
