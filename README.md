# Fake News Detection

## Dataset
WELFake is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. For this, authors merged four popular news datasets (i.e. Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent over-fitting of classifiers and to provide more text data for better ML training.

Dataset contains four columns: 
- **Serial number** - starting from 0 
- **Title** - about the text news heading
- **Text** - about the news content
- **Label** - “Real” is assigned 0 and ‘Fake’ 1.

## Folder Structure
```
fake_news_detection/
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/                 
│   ├── 01_EDA.ipynb           ← exploratory analysis & visualizations  
│   ├── 02_Preprocessing.ipynb ← cleaning, tokenization, splits  
│   ├── 03_Features.ipynb      ← feature engineering (counts, sentiment, LSI/LDA)  
│   ├── 04_Models.ipynb        ← train & evaluate:  
│   │                             • Logistic Regression  
│   │                             • LSTM  
│   │                             • BERT  
│   └── 05_Evaluation.ipynb    ← compare metrics, ROC/PR curves & error analysis  
│
├── src/                       
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models/                
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── lstm.py
│   │   └── bert.py
│   ├── eval.py
│   ├── viz.py
│   └── utils.py
├── reports/
│   ├── figures/
│   └── final_report.md
│
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```