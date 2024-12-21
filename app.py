import os
import re
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained model components
MODEL_PATH = 'models/lstm_chatbot_model.h5'
TOKENIZER_PATH = 'models/lstm_tokenizer.pickle'
LABEL_ENCODER_PATH = 'models/lstm_label_encoder.pickle'
ANSWERS_DICT_PATH = 'models/lstm_answers_dict.json'

# Configuration
max_words = 10000
max_len = 100

# Load saved model components
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(LABEL_ENCODER_PATH, 'rb') as handle:
    label_encoder = pickle.load(handle)

with open(ANSWERS_DICT_PATH, 'r', encoding='utf-8') as f:
    answers_dict = json.load(f)

# Prepare stop words
stop_words = set(stopwords.words('indonesian')) - {'apa', 'bagaimana', 'mengapa', 'kapan', 'dimana', 'siapa'}

def preprocess_text(text):
    """Preprocess input text similar to training data"""
    # Text cleaning
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s*([.,!?])\s*', r' \1 ', cleaned_text)
    cleaned_text = re.sub(r'^\d+\s*', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Tokenization and stop word removal
    tokens = nltk.word_tokenize(cleaned_text)
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        # Get question from request
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({
                'error': 'No question provided',
                'status': 'failed'
            }), 400
        
        # Preprocess the question
        cleaned_question = preprocess_text(question)
        
        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([cleaned_question])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, 
            maxlen=max_len, 
            padding='post'
        )
        
        # Predict category
        prediction = model.predict(padded)
        predicted_index = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_index])
        predicted_category = label_encoder.inverse_transform([predicted_index])[0]
        
        # Find best matching question
        answer = "Maaf, saya tidak dapat menemukan jawaban yang tepat."
        best_match_score = 0
        
        if predicted_category in answers_dict:
            sub_questions = list(answers_dict[predicted_category].keys())
            best_match, best_match_score = process.extractOne(cleaned_question, sub_questions)
            
            if best_match_score >= 80:
                answer = answers_dict[predicted_category][best_match]
        
        # Prepare response
        return jsonify({
            'status': 'success',
            'question': question,
            'category': predicted_category,
            'confidence': confidence,
            'match_score': best_match_score,
            'answer': answer
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/category', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'categories': list(answers_dict.keys())
    }), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from other machines
    app.run(host='0.0.0.0', port=5000, debug=False)