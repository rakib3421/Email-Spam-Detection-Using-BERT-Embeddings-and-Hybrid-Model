from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import torch
from transformers import BertTokenizer, BertModel
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

app = Flask(__name__)

class SpamDetector:
    """Advanced spam detector with BERT and ML ensemble"""
    
    def __init__(self, model_path):
        self.components = None
        self.load_model(model_path)
        
        if self.components:
            self.model = self.components['model']
            self.bert_model = self.components.get('bert_model')
            self.tokenizer = self.components.get('tokenizer')
            self.tfidf_vectorizer = self.components.get('tfidf_vectorizer')
            self.scaler = self.components.get('scaler')
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.bert_model is not None:
                self.bert_model.to(self.device)
                self.bert_model.eval()
    
    def load_model(self, model_path):
        """Load the trained model components"""
        try:
            if os.path.exists(model_path):
                self.components = joblib.load(model_path)
                print(f"Model loaded successfully from {model_path}")
            else:
                print(f"Model file not found: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text using the same logic as training"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' url ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email ', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' phone ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            return ' '.join(tokens)
        except:
            return text
    
    def extract_features(self, text):
        """Extract features for spam detection"""
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'dollar_count': text.count('$'),
            'caps_count': sum(1 for c in text if c.isupper()),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
            'has_url': 1 if 'url' in text.lower() or 'http' in text.lower() else 0,
            'has_email': 1 if '@' in text else 0,
            'has_phone': 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text) else 0,
            'unique_word_ratio': len(set(text.split())) / len(text.split()) if text.split() else 0
        }
        
        # Spam indicators
        spam_indicators = [
            'free', 'win', 'winner', 'cash', 'prize', 'urgent', 'limited', 'offer', 'deal', 'discount',
            'guarantee', 'money', 'credit', 'loan', 'investment', 'million', 'billion', 'rich', 'wealth',
            'congratulations', 'congrats', 'claim', 'click', 'subscribe', 'unsubscribe', 'buy', 'sale',
            'marketing', 'advertisement', 'promotion', 'newsletter', 'update', 'alert', 'notification'
        ]
        
        features['spam_word_count'] = sum(1 for word in spam_indicators if word in text.lower())
        features['spam_word_ratio'] = features['spam_word_count'] / features['word_count'] if features['word_count'] > 0 else 0
        
        return features
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for text"""
        if self.bert_model is None or self.tokenizer is None:
            return np.zeros(768)
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            return embedding
        except:
            return np.zeros(768)
    
    def predict(self, subject, message):
        """Make spam prediction"""
        if not self.components:
            return {
                'prediction': 'error',
                'spam_probability': 0.0,
                'ham_probability': 0.0,
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Combine subject and message
            full_text = subject + ' ' + message
            processed_text = self.preprocess_text(full_text)
            
            # Extract features
            features = self.extract_features(full_text)
            
            # Get BERT embedding
            bert_embedding = self.get_bert_embedding(processed_text)
            
            # Get TF-IDF features
            if self.tfidf_vectorizer is not None:
                tfidf_features = self.tfidf_vectorizer.transform([processed_text]).toarray()[0]
            else:
                tfidf_features = np.zeros(5000)
            
            # Get numerical features
            numerical_features = np.array([
                features['char_count'], features['word_count'], features['sentence_count'],
                features['avg_word_length'], features['exclamation_count'], features['question_count'],
                features['dollar_count'], features['caps_count'], features['caps_ratio'],
                features['spam_word_count'], features['spam_word_ratio'], features['has_url'],
                features['has_email'], features['has_phone'], features['unique_word_ratio']
            ])
            
            # Scale numerical features
            if self.scaler is not None:
                numerical_features_scaled = self.scaler.transform(numerical_features.reshape(1, -1))[0]
            else:
                numerical_features_scaled = numerical_features
            
            # Combine all features
            combined_features = np.concatenate([
                bert_embedding,
                tfidf_features,
                numerical_features_scaled
            ])
            
            # Make prediction
            prediction = self.model.predict(combined_features.reshape(1, -1))[0]
            probabilities = self.model.predict_proba(combined_features.reshape(1, -1))[0]
            
            return {
                'prediction': 'spam' if prediction == 1 else 'ham',
                'spam_probability': float(probabilities[1]),
                'ham_probability': float(probabilities[0]),
                'confidence': float(max(probabilities)),
                'processed_text': processed_text,
                'features': features,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'prediction': 'error',
                'spam_probability': 0.0,
                'ham_probability': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

# Initialize the spam detector
detector = SpamDetector('spam_detector_components.pkl')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle spam prediction requests"""
    try:
        data = request.get_json()
        subject = data.get('subject', '')
        message = data.get('message', '')
        
        if not subject.strip() and not message.strip():
            return jsonify({
                'error': 'Please provide either a subject or message to analyze'
            }), 400
        
        result = detector.predict(subject, message)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get model statistics"""
    return jsonify({
        'model_info': {
            'name': 'BERT + ML Ensemble Spam Detector',
            'precision': '98.7%',
            'recall': '97.7%',
            'f1_score': '98.1%',
            'auc': '99.96%',
            'training_samples': '5,854 emails'
        },
        'features': [
            'BERT Transformer embeddings',
            'TF-IDF text features',
            'Statistical text analysis',
            'Spam keyword detection',
            'URL and email pattern recognition',
            'Ensemble ML models (SVM, RF, XGBoost)'
        ]
    })

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
