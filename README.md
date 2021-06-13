# DSAI-HW4-2021

### Introduction
Import the requirements using `pip install -r requirements.txt`.

Note that due to the limit of bandwidth and storage on Git LFS, the required datasets are additionally provided via [Google Drive](https://drive.google.com/drive/folders/1n-VnQg3oDkyGPrLt4e8uoWJiD9b6cHhm?usp=sharing) instead of GitHub.  
To run the code, download the [zipped data files](https://drive.google.com/file/d/1L0VMO67wquwywR_a6SQ37Wf0A77yojRw/view?usp=sharing) and unzip it under the `data` folder. 
Then run `python xgboost_model.py`, and `submission.csv` will be saved in the root folder.

The details are in [Google Doc report](https://docs.google.com/document/d/1hkh-fauw2Un097nzzFQIOhN7blrB9jlRYbpEDMKpk6M/edit?usp=sharing).

### Project Organization
```
.
├── data/                                       : Stores datasets
├── image/                                      : Contains all plots 
├── data_description.py                         : Initial analysis to understand data
├── market_basket_analysis.py                   : Market Basket Analysis to find products association
├── feature_extraction.py                       : Feature engineering and extraction for a ML model
├── data_preparation.py                         : Data preparation for modeling
├── xgboost_model.py                            : XGBoost model for product reorder prediction
├── model.txt                                   : Save trained model
├── submission.csv                              : Output results
└── README.md                                   : Project introduction 
```