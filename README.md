# News Article Classification (NLP)

An Applied Machine Learning project implementing a Natural Language Processing (NLP) pipeline to categorize news articles into four classes: **Sports, World, Business, and Sci/Tech**.

## Project Overview
This study performs a comparative analysis between a probabilistic model (**Multinomial Naive Bayes**) and a deep learning architecture (**Bi-Directional LSTM**) using the standardized AG’s News Corpus.

---

## Technical Pipeline



### 1. Data Preprocessing
To ensure high-quality input for the models, the following normalization steps were applied:
* **Text Cleaning:** Lowercasing, removal of special characters, and publisher-specific tags.
* **Contraction Expansion:** Converting "don't" to "do not" using the `contractions` library.
* **Stop-word Removal:** Filtering out common grammatical words (e.g., "the", "is").
* **Lemmatization:** Reducing words to their root form (e.g., "running" to "run") via NLTK.

### 2. Feature Extraction
* **For Naive Bayes:** **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
* **For Bi-LSTM:** **GloVe (Global Vectors)** pre-trained word embeddings combined with sequence padding (max length of 100 tokens).

### 3. Model Architectures
* **Multinomial Naive Bayes:** A high-efficiency probabilistic model based on Bayes’ Theorem.
* **Bi-Directional LSTM:** A recurrent neural network (RNN) that captures dependencies in both forward and backward directions to understand context.



---

## Performance Results

| Model | Test Accuracy | Characteristics |
| :--- | :--- | :--- |
| **Bi-Directional LSTM** | **90.24%** | Higher accuracy; captures complex context. |
| **Multinomial Naive Bayes** | **89.73%** | Superior computational speed and efficiency. |

### Evaluation Metrics
Both models achieved high scores (~0.90) in **Precision, Recall, and F1-Score**. While Bi-LSTM had the edge in raw accuracy, Naive Bayes proved highly effective for low-latency requirements.

---

## Tools & Technologies
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Machine Learning:** Scikit-Learn
* **NLP Libraries:** NLTK, Contractions
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---
