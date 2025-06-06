# Fake News Detection

This repository provides a comprehensive pipeline for detecting fake news using machine learning and deep learning techniques. It leverages the WELFake dataset, which combines multiple sources to ensure robust model training and evaluation.

## Dataset

**WELFake** consists of 72,134 news articles, with 35,028 labeled as real and 37,106 as fake. The dataset merges four popular sources (Kaggle, McIntire, Reuters, BuzzFeed Political) to enhance diversity and reduce overfitting.

Each entry contains:
- **Serial number**: Unique identifier
- **Title**: News headline
- **Text**: News content
- **Label**: 0 for real, 1 for fake

## Project Structure

```
fake_news_detection/
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned and prepared data
│
├── notebooks/
│   ├── 1-EDA.ipynb           # Exploratory data analysis & visualization
│   ├── 2-Preprocessing.ipynb # Data cleaning, tokenization, splitting
│   ├── 3.1-LogReg.ipynb      # Logistic Regression: feature engineering & training
│   ├── 3.2-LSTM.ipynb        # LSTM: model training & evaluation
│   ├── 3.3-BERT.ipynb        # BERT: model training & evaluation
│   └── 4-ExternalVal.ipynb   # External validation & final metrics
│
├── models/
│   ├── logreg/               # Saved Logistic Regression models and artifacts
│   ├── lstm/                 # Saved LSTM models and checkpoints
│   └── bert/                 # Saved BERT models and checkpoints
│
├── src/
│   ├── plot.py               # Functions for generating plots and visualizations
│   └── utils.py              # Utility functions for data processing and model support
│
├── reports/
│   └── report.pdf            # Project summary and findings
│
├── setup.py                  # Package setup
├── README.md
├── LICENSE
└── .gitignore
```

## Getting Started

1. **Clone the repository**
    ```bash
    git clone https://github.com/Jaegon99/fake-news-detection.git
    cd fake_news_detection
    pip install .  
    ```
2. **Explore the notebooks**
    - Start with `01_EDA.ipynb` for data exploration.
    - Proceed through preprocessing, feature engineering, modeling, and evaluation.

## Models Implemented

- **Logistic Regression**
- **LSTM (Long Short-Term Memory)**
- **BERT (Bidirectional Encoder Representations from Transformers)**

## Results

Evaluation metrics and analysis are available in the `reports/report.pdf` or look directly in a model notebook.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Acknowledgements

- WELFake dataset authors
- [Neptune.ai. (2021). Exploratory Data Analysis for Natural Language Processing: Tools and Techniques.](https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools)
- [mexwell. (2021). BERT for Binary Classification. Kaggle Notebook.](https://www.kaggle.com/code/mexwell/bert-for-binary-classification/notebook)
- [emineyetm. (2021). Fake News Detection Datasets. Kaggle Dataset.](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
