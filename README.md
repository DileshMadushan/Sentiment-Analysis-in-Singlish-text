# Sentiment Analysis in Singlish Text

This repository contains the research project **"Sentiment Analysis in Singlish Text"** conducted as part of our final year undergraduate research. The project focuses on building a sentiment classification model that can accurately detect sentiment polarity in **Singlish — a unique blend of Sinhala and English frequently used in digital communication in Sri Lanka**.

## 🧠 Project Overview

Traditional sentiment analysis models often perform poorly on informal and non-standard language like Singlish due to lack of suitable datasets and linguistic complexity. Our research project addresses this gap by:

- Creating a **custom Singlish sentiment dataset** from Sri Lankan online sources such as product reviews and social media posts.
- Preprocessing Singlish text with tokenization, normalization, and lemmatization.
- Experimenting with traditional ML models (e.g., Naive Bayes, Logistic Regression, SVM).
- Evaluating model performance using standard classification metrics such as Accuracy, Precision, Recall, and F1-score.

## 📊 Dataset

A custom Singlish review dataset was created as part of this research. It includes:
- User-generated reviews collected from local platforms.
- Manual labeling for sentiment classification.
- Preprocessing steps including tokenization, normalization, and handling of Singlish-specific expressions.

## 🧠 Methodology

The following steps were followed in the project:

1. **Data Collection** – Customer reviews were gathered from local forums, product review pages, and social media platforms.
2. **Data Preprocessing** – Included text cleaning, removal of noise, and conversion of Singlish expressions to standard English equivalents where appropriate.
3. **Sentiment Labeling** – Each entry was labeled as Positive, Negative, or Neutral manually or using heuristics.
4. **Model Training** – Several models were tested:
   - Traditional ML models: Naive Bayes, Logistic Regression, SVM
   - Deep Learning: LSTM and BERT (fine-tuned with multilingual pre-trained models)
5. **Evaluation** – The models were evaluated using accuracy, precision, recall, and F1-score.

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Scikit-learn
- TensorFlow / PyTorch
- Hugging Face Transformers
- NLTK / spaCy
- Pandas, NumPy
