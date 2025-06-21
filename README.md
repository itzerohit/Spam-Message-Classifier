# Spam-Message-Classifier
SMS Spam Prediction project using data cleaning, EDA, text preprocessing, and classification.

📩 Spam Message Classifier using Machine Learning
This project is a Spam Detection System that classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and various Machine Learning algorithms. It is designed as a complete data science workflow — from data cleaning to model evaluation and deployment.

🚀 Key Features
📊 Exploratory Data Analysis (EDA) to understand patterns in text and class distributions

🧹 Text Preprocessing: Lowercasing, tokenization, stopword removal, stemming

🔠 Feature Extraction using TF-IDF Vectorization

🤖 Model Building with:

Multinomial Naive Bayes

Logistic Regression

Support Vector Machines (SVM)

Random Forest

XGBoost

Ensemble models (Voting, Stacking)

📈 Model Evaluation using Accuracy, Precision, and Confusion Matrix

🌐 Web Deployment-Ready: The model is serialized and can be deployed with Streamlit or Flask

📁 Dataset Used
Source: UCI SMS Spam Collection Dataset

Total Records: 5,572 SMS messages labeled as spam or ham

🧪 Evaluation Results
Model	Accuracy	Precision
Naive Bayes	97.1%	100.0%
Logistic Regression	96.1%	97.1%
Random Forest	97.4%	98.2%
XGBoost	97.1%	95.0%
SVM (sigmoid)	97.2%	97.4%
Ensemble (Voting)	98.1%	99.1%

📦 Libraries Used
pandas, numpy

matplotlib, seaborn

sklearn

nltk

xgboost

pickle

🛠️ How It Works
Load and clean the dataset

Perform EDA and visualize spam vs ham distributions

Preprocess text using NLP techniques

Convert text into numeric form using TF-IDF

Train various ML models

Evaluate performance metrics

Export the best model using pickle for deployment

💡 Future Work
Add real-time prediction web app using Streamlit

Improve accuracy using deep learning models (e.g., LSTM, BERT)

Integrate email/SMS API for real-time spam filtering
