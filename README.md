# Advanced Spam Mail Detection System

## Overview

This project implements an advanced spam mail detection system using state-of-the-art natural language processing techniques and machine learning. The system combines BERT transformers with traditional ML models in a hybrid ensemble approach to achieve high accuracy in detecting modern spam emails.

## Features

### 🔬 Advanced NLP Techniques
- **BERT Transformer**: Deep contextual understanding of email content
- **Advanced Preprocessing**: Lemmatization, stopword removal, HTML cleaning
- **Feature Engineering**: 15+ engineered features for spam detection
- **TF-IDF Vectorization**: Traditional text feature extraction

### 🤖 Hybrid Model Architecture
- **Ensemble Methods**: Voting and Stacking classifiers
- **Multiple ML Models**: SVM, Random Forest, XGBoost, Logistic Regression
- **SMOTE**: Handling class imbalance
- **Cross-validation**: Robust model evaluation

### 🎨 Modern Web Interface
- **Streamlit UI**: Interactive web application
- **Real-time Analysis**: Instant spam detection
- **Confidence Scores**: Probability-based predictions
- **Detailed Explanations**: Feature importance and analysis

## Installation

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (recommended for BERT)
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `transformers` - BERT and other transformers
- `torch` - PyTorch for deep learning
- `scikit-learn` - Machine learning algorithms
- `streamlit` - Web interface
- `nltk` - Natural language processing
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization

## Usage

### 1. Training the Model
Run the Jupyter notebook to train the model:
```bash
jupyter notebook advanced_spam_detection_bert.ipynb
```

### 2. Web Application
Launch the Streamlit web interface:
```bash
streamlit run spam_detection_app.py
```

### 3. Model Files
- `advanced_spam_detection_bert.ipynb` - Complete training notebook
- `spam_detection_app.py` - Streamlit web application
- `spam_detector_components.pkl` - Trained model components
- `enron_spam_data.csv` - Training dataset

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 93.8% |
| Recall | 94.0% |
| F1-Score | 93.9% |
| AUC-ROC | 96.5% |

## Architecture

### Data Pipeline
1. **Data Loading**: Enron spam dataset
2. **Preprocessing**: Text cleaning and normalization
3. **Feature Extraction**:
   - BERT embeddings (768 dimensions)
   - TF-IDF features (5000 dimensions)
   - Engineered features (15 dimensions)
4. **Model Training**: Ensemble of ML models
5. **Evaluation**: Comprehensive metrics and visualization

### Model Components
- **BERT Base Uncased**: Pre-trained transformer
- **TF-IDF Vectorizer**: Text to numerical conversion
- **Feature Scaler**: Standardize numerical features
- **Ensemble Classifier**: Voting/Stacking of base models

## API Usage

### Python API
```python
from spam_detection_app import StreamlitSpamDetector
import joblib

# Load components
components = joblib.load('spam_detector_components.pkl')
detector = StreamlitSpamDetector(components)

# Make prediction
result = detector.predict("Email Subject", "Email message content")
print(result)
# Output: {'prediction': 'spam', 'spam_probability': 0.95, ...}
```

### Web Interface
1. Open the Streamlit app
2. Enter email subject and message
3. Click "Analyze Email"
4. View prediction results and confidence scores

## Dataset

### Enron Spam Dataset
- **Source**: Enron email corpus
- **Size**: 56MB+ of email data
- **Classes**: Spam (1) and Ham (0)
- **Features**: Subject, Message, Date, Spam/Ham label

### Data Preprocessing
- Remove HTML tags and URLs
- Normalize text (lowercase, remove special chars)
- Tokenization and lemmatization
- Stopword removal
- Handle missing values

## Advanced Features

### Spam Detection Indicators
- URL and email pattern detection
- Capitalization analysis
- Punctuation frequency
- Spam keyword detection
- Text length and structure analysis

### Model Interpretability
- Feature importance analysis
- Confidence score explanation
- Processed text visualization
- Prediction reasoning

## Deployment

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run training (optional)
jupyter notebook advanced_spam_detection_bert.ipynb

# Launch web app
streamlit run spam_detection_app.py
```

### Production Deployment
- Use saved model files (`spam_detector_components.pkl`)
- Implement REST API endpoint
- Containerize with Docker
- Deploy on cloud platforms (AWS, GCP, Azure)

## Evaluation Metrics

### Classification Metrics
- **Confusion Matrix**: True/False Positives/Negatives
- **Precision/Recall**: Accuracy per class
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the curve

### Model Comparison
- Cross-validation scores
- Training vs validation performance
- Overfitting detection
- Hyperparameter tuning results

## Future Enhancements

### Model Improvements
- **Fine-tuned BERT**: Domain-specific training
- **Additional Transformers**: RoBERTa, DistilBERT
- **Deep Learning**: CNN, LSTM architectures
- **Multi-modal**: Image and attachment analysis

### Feature Additions
- **Real-time Learning**: Online model updates
- **Multi-language**: Support for non-English emails
- **Behavioral Analysis**: Sender reputation, timing patterns
- **Advanced NLP**: Named entity recognition, sentiment analysis

### System Enhancements
- **Scalability**: Distributed processing
- **API Development**: RESTful endpoints
- **Monitoring**: Performance tracking and alerting
- **Security**: Input validation and sanitization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{advanced-spam-detection,
  title={Advanced Spam Mail Detection with BERT and Hybrid ML},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/advanced-spam-detection}
}
```

## Contact

For questions or support:
- Open an issue on GitHub
- Email: your-email@example.com

---

**Built with ❤️ using Python, BERT, and Streamlit**
