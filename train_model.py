# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:14:02 2024

@author: Abdelrahman
"""

# train_model.py

import re
#
import pyarabic.araby as araby
from pyarabic.araby import strip_tashkeel
from camel_tools.tokenizers.word import simple_word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def process_arabic_text(text):
    """Processes Arabic text by correcting spelling mistakes first, then removing emojis, diacritics,
       normalizing hamzas, tokenizing, removing stop words, punctuation, non-Arabic word removal, and stemming.
    Args:
        text: The Arabic text to process and correct.

    Returns:
        The corrected Arabic sentence.
    """
##
    
    # Emoji removal
    emoji_pattern = re.compile(pattern = "["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
    corrected_text = emoji_pattern.sub(r"", corrected_text)

    # Normalization
    normalized_text = strip_tashkeel(corrected_text)  # Remove diacritics
    normalized_text = araby.normalize_hamza(normalized_text)  # Normalize hamzas
    normalized_text = normalized_text.replace('ي', 'ى')  # Alef maksura to ya
    normalized_text = normalized_text.replace('ء','ا')  # Alef with hamza to alef
    normalized_text = normalized_text.replace('ة', 'ه')  # Teh marbuta to ta marbuta

    # Punctuation Removal
    punctuations = "!؟؛،«»\\,\\:\\;\\(\\)\\(//)\\-\\_\\~\\#\\@\\$\\%\\[\\]\\{\\}\\+\\|\\*\\=\\<\\>\\^\\&\\٪"  # Arabic punctuation marks
    normalized_text = re.sub(f"[{punctuations}]", "", normalized_text)

    # Tokenization
    tokens = simple_word_tokenize(normalized_text)  # Split into words

    # Stop word removal
    stop_words = set(stopwords.words("arabic"))  # Load Arabic stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = SnowballStemmer("arabic")  # Use Arabic-specific stemmer
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    stemmed_text = ' '.join(stemmed_tokens).rstrip('.?!').lstrip('.?!')

    return stemmed_text


# Read the xlsx file
df = pd.read_excel('complains 2.xlsx')

# Convert the dataframe to CSV
df.to_csv('complains 2.csv', index=False)

# Label Encoding
le = LabelEncoder()
le.fit(df['تصنيف الشكوى'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
y_labeled = le.transform(df['تصنيف الشكوى'])

df['تصنيف الشكوى_معالج'] = le.transform(df['تصنيف الشكوى'])

# Preprocess the text
X = df['مضمون الشكوى'].apply(process_arabic_text)
df['مضمون الشكوى'] = X

# Oversample minority class using SMOTE
smote = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=1)
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_resampled, y_resampled = smote.fit_resample(X_vec, y_labeled)

# Train the MLP classifier
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=44, shuffle=True)

mlp = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(128, 64), random_state=42, max_iter=500)
mlp.fit(X_train, y_train)

# Save the trained model and other necessary objects
joblib.dump(mlp, 'saved_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("MLP classifier model trained and saved successfully.")
