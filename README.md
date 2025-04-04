#  Arabic-NLP / complain-system

```markdown
# Arabic Complaint Classification System

A machine learning system for classifying Arabic complaints using text processing and neural networks.

## Features

- Arabic text preprocessing (normalization, tokenization, stemming)
- Complaint classification using MLP neural network
- REST API endpoint for predictions
- Handles class imbalance with SMOTE

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdelrahansaid/arabic-complaint-classification.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- pyarabic
- camel-tools
- flask (for API)
- imbalanced-learn

## Usage

### Training the Model
Run the training script:
```bash
python train_model.py
```

### Starting the API
```bash
python api.py
```

### Making Predictions
Send a POST request to `/classify` with JSON payload:
```json
{
    "complaint_text": "نص الشكوى هنا"
}
```

## Text Processing Pipeline

1. Emoji removal
2. Diacritics removal
3. Hamza normalization
4. Punctuation removal
5. Tokenization
6. Stopword removal
7. Stemming

## Model Details

- **Algorithm**: Multi-layer Perceptron (MLP)
- **Vectorization**: TF-IDF
- **Class Balancing**: SMOTE oversampling
- **Hidden Layers**: 128, 64

## Files

- `train_model.py` - Training script
- `api.py` - Flask REST API
- `saved_model.pkl` - Trained model
- `vectorizer.pkl` - TF-IDF vectorizer
- `label_encoder.pkl` - Label encoder

## API Endpoints

- `POST /classify` - Classify complaint text

Example response:
```json
{
    "predicted_label": "فئة الشكوى"
}
```

## Author

Abdelrahman Said Mohamed

www.linkedin.com/in/abdelrahman-said-mohamed-96b832234

## License

[MIT](LICENSE)
