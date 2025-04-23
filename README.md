# AraComplaintClassifier

A Python-based NLP pipeline to preprocess and classify Arabic complaints using a neural network, with a Flask REST API for real-time prediction.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Data Description](#data-description)  
4. [Processing & Modeling Pipeline](#processing--modeling-pipeline)  
5. [Usage](#usage)  
6. [Schema & Performance](#schema--performance)  
7. [Results](#results)  
8. [Future Work](#future-work)  
9. [Author](#author)  
10. [License](#license)  

---

## Project Overview

**AraComplaintClassifier** is an end-to-end machine-learning system for automatically categorizing Arabic customer complaints. It includes text normalization, feature extraction (TF-IDF), MLP classification, and a Flask API endpoint for serving predictions.

---

## Motivation

Organizations handling large volumes of Arabic text complaints need an automated way to route and prioritize issues. This system accelerates response times, improves accuracy of categorization, and scales to high throughput via a simple REST interface.

---

## Data Description

- **Source**: Internal or publicly collected Arabic complaint texts  
- **Records**: N complaint examples, each labeled with one of K categories  
- **Fields**:  
  - `complaint_text` (string)  
  - `label` (category)  

---

## Processing & Modeling Pipeline

1. **Text Preprocessing**  
   - Emoji removal  
   - Diacritics and punctuation removal  
   - Hamza normalization  
   - Tokenization  
   - Stopword removal  
   - Light stemming  
2. **Feature Extraction**  
   - TF-IDF vectorization of cleaned tokens  
3. **Handling Class Imbalance**  
   - SMOTE oversampling on minority classes  
4. **Model Architecture**  
   - Multi-Layer Perceptron (MLP) with hidden layers [128, 64]  
   - ReLU activations, softmax output  
5. **Training & Validation**  
   - 80/20 train-test split  
   - Metrics: accuracy, precision, recall, F1-score  

---

## Usage

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Abdelrahansaid/arabic-complaint-classification.git
   cd arabic-complaint-classification




   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model** (optional if using provided artifacts)  
   ```bash
   python train_model.py
   ```
4. **Start the API server**  
   ```bash
   python api.py
   ```
5. **Classify a complaint**  
   Send a POST to `http://localhost:5000/classify` with JSON:  
   ```json
   {
     "complaint_text": "نص الشكوى هنا"
   }
   ```
   Response:
   ```json
   {
     "predicted_label": "فئة_الشكوى"
   }
   ```

---

## Schema & Performance

- **TF-IDF vocabulary size**: ~10,000 terms  
- **Model parameters**: ~200K weights  
- **Inference time**: <20 ms per request  
- **Throughput**: ~50 requests/sec on a single CPU core  

---

## Results

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 96 %   |
| Precision | 96.3%   |
| Recall    | 96.5%   |
| F1-Score  | 96.39%   |

---

## Future Work

- Experiment with transformer-based embeddings (AraBERT)  
- Deploy as Docker container or serverless function  
- Add multi-label support for complex complaints  
- Create a web-UI for non-technical users  

---

## Author

**Abdelrahman Said Mohamed**  
- 📎 [LinkedIn](https://www.linkedin.com/in/abdelrahman-said-mohamed-96b832234/)  
- ✉️ abdelrahmanalgamil@gmail.com  

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  
