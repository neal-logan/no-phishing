# No Phishing: Detecting Malicious URLs

### Overview

This project covers the development of phishing URL detection models.  Given a set of features extracted from the URLs themselves, generated from the corresponding sites, or obtained from third parties, these models predict whether the URL in question is legitimate or used for phishing. 

It began as a [project](https://github.com/neal-logan/dsba6211-summer2024/tree/main/nophishing) at UNC Charlotte (in Advanced Business Analytics, a fantastic class taught by [Ryan Wesslen](https://github.com/wesslen)) which I attended in Summer 2024.

### Table of Contents

[01 Exploratory Analysis](https://github.com/neal-logan/dsba6211-summer2024/blob/main/nophishing/01_exploratory_analysis.ipynb) - This notebook covers extensive exploratory analysis of the dataset, including some preliminary modeling used to identify the most relevant features.  

[02 Modeling Notebook](https://github.com/neal-logan/dsba6211-summer2024/blob/main/nophishing/02_modeling.ipynb) - This notebook includes core modeling efforts on features selected in the Exploratory Analysis notebook, including the development of the final model.

[Training Data](https://raw.githubusercontent.com/neal-logan/dsba6211-summer2024/main/nophishing/data/phishing-url-pirochet-train.csv)

[Test Data](https://raw.githubusercontent.com/neal-logan/dsba6211-summer2024/main/nophishing/data/phishing-url-pirochet-test.csv) 

[requirements.txt](https://github.com/neal-logan/dsba6211-summer2024/blob/main/nophishing/requirements.txt) - This document identifies packages (including versions) directly imported or used in the project. However, it does not identify those packages' requirements, which may be numerous.

### Reproducing Results

The results can be reproduced by simply running the Colab notebook [02 Modeling](https://github.com/neal-logan/dsba6211-summer2024/blob/main/nophishing/02_modeling.ipynb).  This notebook will install and import the necessary packages of the correct versions, load and transform the [data](https://github.com/neal-logan/dsba6211-summer2024/tree/main/nophishing/data), run and evaluate the models, and finally explain key features of the models and the data itself.  The random seed is embedded in the notebook and provided to processes wherever necessary to obtain consistent results.

## Introduction

Malicious URLs are a common component of phishing attacks.  They are sometimes used to exploit technical vulnerabilities, executing malicious code  automatically when the message is presented to the target, when the target interacts with the message, or when the target follows the link to the malicious URL.  However, attacks relying primarily on social engineering present a more difficult challenge, for example by leading phishing targets to sites that appear entirely legitimate even to relatively vigilant and sophisticated internet users.  Detecting these malicious URLs provides us with several opportunities for defense, and is an important part of engineering secure systems.

Malicious URL detection can be used in several ways.  It can be used to provide suspected-phishing warnings to users, particularly in web browsers or email/messaging systems.  It can also be used by organizations, which can warn or block users from visiting suspected phishing sites.  And finally, malicious URL detection can be used by to identify or refer malicious sites for takedowns or to help direct law enforcement efforts against the threat actors behind the malicious sites.

## Literature Review

####


#### [Phishing URL Detection](https://github.com/pirocheto/phishing-url-detection) by Pirocheto

This repository contains a complete project for phishing URL detection using machine learning and MLOps practices. It uses a TF-IDF vectorizer using both character and word n-grams) with a linear SVM model. The code is designed to be lightweight and fast, suitable for embedding in applications, and can work offline, without an internet connection. The repository also includes instructions for reproducing the model and running the pipeline.

#### [PhishShield](https://github.com/praneeth-katuri/PhishShield) by Praneeth Katuri

This GitHub repository provides a comprehensive solution for detecting phishing websites using analytical models and custom transformers for preprocessing. It includes feature-based and text-based models, including random forest, LGBM, SVC, logistic regression, and Multinomial Naive Bayes, and takes advantage of grid-search with cross-validation. The repository also offers Flask deployment for real-time URL prediction and caching for performance improvement.

#### [Phishing Link Detection](https://github.com/Sayan-Maity-Code/Phishing-link-detection) by Sayan Maity

This project uses Multinomial Naive Bayes and Logistic Regression to detect malicious URLs. The model's preprocessing involves tokenization and TF-IDF vectorization. The project includes scripts for training and evaluating the model.

## Dataset

The data was obtained from [HuggingFace](https://huggingface.co/datasets/pirocheto/phishing-url).  Please see data analysis and preparation sections of the Exploratory Analysis and Modeling notebooks for further detail.

## Quick Summary

In **Exploratory Analysis**:
* Missing/erroneous data was identified in two features, and preprocessing steps were developed to address the issue, and
* 14 features were selected based on preliminary modeling and feature importance analysis.

In **Modeling**: 
* These 14 selected features and the preprocessing steps developed to address the missing/erroneous were implemented to develop a new series of models, using the training data set above split into a smaller training set and a validation set;
* A new model was developed with monotonicity constraints applied to five features, based on concerns identified in partial dependency plots;
* The development models were assessed for accuracy, overfitting, weakspots, and resilience; and
* A final model was developed and tested for accuracy against the test dataset.

## Conclusions & Future Directions

**Probably a pretty good model**: The final model's ROC-AUC, precision, and recall are all about 0.96.

**However, it's nowhere near ready for production**:
* The data more or less makes sense based on my experience, but I've done no verification of the data quality, and in particular haven't done anything to verify that it is related to any real-world data
* The final model has not been assessed for overfitting, weakspots, or resilience, although there's little reason to expect these concerns to differ from the exploratory models
* The model is heavily reliant on a few external services
* PiML's ROC-AUC seemed to disagree substantially with sklearn's ROC-AUC calcuation on models that were probably identical; I reported the more-realistic 0.96 value from sklearn rather than the 0.99 calculated by PiML. 
* Regularization was not applied, although since the included parameters are limited this may not be much of a problem
* No cross-validation/hyperparameter search was used to optimize parameters
* There's no consideration for exporting models
* There's probably room for some feature engineering


## Some Lessons Learned
* **Use PiML early and often**, particularly in **early** exploratory data analysis. I intended to use PiML extensively in EDA, but only for things that I didn't end up getting to. In fact, it would have produced better results faster for many of the analyses I did complete in this section.
* **Don't use Colab** (or any web-based notebook) for processes that take more than a couple of minutes.  While Colab is convenient for quick notebooks, efforts of even this fairly scale can benefit from more control over the environment, more (and more consistent) compute power, and most importantly less disruption from connection instability.
