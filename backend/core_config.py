### main.py
VERBOSE = True
DATASET = {
    # 'name': ''
    'path': 'credit.xlsx',
    'target': 'default payment next month',
    'protected': ['SEX', 'MARRIAGE'],
}

DATASET_INFO = {
    "credit": {
        "data": 'credit.xlsx',
        'attrs': [
            'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
            'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX', 'payment'
        ],
        'num_attrs': [
            'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ],
        'cate_attrs': [
            'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX', 'payment'
        ]
    }
}

# Parameters for converting numerical protected attributes to categorical variables
PARAMS_NUM_TO_CAT_METHOD = 'quartile'  # 'median' or 'quartile'
PARAMS_NUM_TO_CAT_CUTS = 4  # Number of cuts when using custom binning

SEED = 0
USE_BIAS_MITIGATION = True
USE_ACCURACY_ENHANCEMENT = False
PARAMS_MAIN_STEP = 'd3B'
# Supported classifier types: LR(Logistic Regression), DT(Decision Tree), KNN(K-Nearest Neighbors), GBDT(Gradient Boosting Decision Tree),
# ADABoost(AdaBoost), NB(Naive Bayes), SVM(Support Vector Machine), MLP(Multi-layer Perceptron),
# XGBoost(eXtreme Gradient Boosting), RF(Random Forest), LGBM(LightGBM),
# CatBoost, LDA(Linear Discriminant Analysis), QDA(Quadratic Discriminant Analysis)
PARAMS_MAIN_CLASSIFIER = 'LR'
PARAMS_MAIN_MAX_ITERATION = 2
PARAMS_MAIN_TRAINING_RATE = 0.5
PARAMS_MAIN_THRESHOLD_EPSILON = 0.9
PARAMS_MAIN_THRESHOLD_ACCURACY = 0.01
PARAMS_MAIN_AE_IMPORTANCE_MEASURE = 'a1'
PARAMS_MAIN_AE_REBIN_METHOD = 'r1'
PARAMS_MAIN_ALPHA_O = 0.8


### eval.py
PARAMS_EVAL_H_ORDER = 'default'  # 1- N
PARAMS_EVAL_SUM = 'd1A'  # d1A, d1B
PARAMS_EVAL_CAT = 'cat-a'  # a, b
PARAMS_EVAL_NUM = 'num-a'  # a, b, c, d
PARAMS_EVAL_SCALE = 'zscore' # mean, min, zscore
PARAMS_EVAL_DIST_METRIC = 'euclidean'
"""
braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulczynski1, mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule
"""
# List of supported fairness metrics:
# BNC: Between Negative Classes
# BPC: Between Positive Classes
# CUAE: Conditional Use Accuracy Equality
# EOpp: Equal Opportunity
# EO: Equalized Odds
# FDRP: False Discovery Rate Parity
# FORP: False Omission Rate Parity
# FNRB: False Negative Rate Balance
# FPRB: False Positive Rate Balance
# NPVP: Negative Predictive Value Parity
# OAE: Overall Accuracy Equality
# PPVP: Positive Predictive Value Parity
# SP: Statistical Parity
PARAMS_EVAL_METRIC_FAIRNESS = ['BNC', 'BPC', 'CUAE', 'EOpp', 'EO', 'FDRP', 'FORP', 'FNRB', 'FPRB', 'NPVP', 'OAE', 'PPVP', 'SP']
# List of supported accuracy metrics:
# ACC: Accuracy
# F1: F1 Score - harmonic mean of precision and recall
# Recall: True Positive Rate - proportion of positive cases correctly identified
# Precision: Positive Predictive Value - proportion of predicted positives that are actual positives
PARAMS_EVAL_METRIC_ACCURACY = ['ACC', 'F1', 'Recall', 'Precision']


### transform.py
PARAMS_TRANSFORM = 'poly'  # poly, log, arcsin
PARAMS_TRANSFORM_MULTI = 't1'  # t1, t2, t3
PARAMS_TRANSFORM_STREAM = 'd4A'  # d4A, d4B, E1-9
PARAMS_TRANSFORM_STREAM_CONFIG = {
    'p': 102,
    'q': 173,
    'emin': 1/2,
    'emax': 2,
    'order': 0,
    'length': 10
}


