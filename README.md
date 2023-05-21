# Gesture Recognition Models

This repository contains implementations of two gesture recognition models: DTW (Dynamic Time Warping) and 1$ Recognizer. These models are designed to classify and recognize gestures based on time series data.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Evaluation](#evaluation)

## Introduction

Gesture recognition is an important area in human-computer interaction, enabling devices to understand and interpret human gestures for various applications. The DTW and 1$ Recognizer models provided in this repository offer different approaches to gesture recognition.

- The DTW model utilizes Dynamic Time Warping as the distance measure between pairs of time series data. It calculates the similarity between a test sample and training samples and uses k-nearest neighbors to predict the class label.

- The 1$ Recognizer model is a gesture recognition algorithm based on the $P (one-dollar) gesture recognition system. It represents gestures as sequences of points and compares them using geometric matching techniques.


## Usage

The usage of the gesture recognition models involves the following steps:

1. Prepare your gesture dataset and labels. The repository provides two datasets from the study conducted by Huang et al. (2019).

2. Import the necessary classes and functions from the respective model files (`DTW.py` and `OneDollar.py`) and the `shared_func` module.

3. Create an instance of the chosen model (DTW or 1$ Recognizer).

4. Preprocess the dataset, if required, using the functions provided in the `shared_func` module. This may include standardization, PCA variance, or other data preprocessing techniques.

5. Split the dataset into training and test sets using user-dependent or user-independent cross-validation methods available in the `shared_func` module.

6. Fit the model on the training data using the `fit()` method.

7. Make predictions on the test data using the `predict()` method.

8. Evaluate the performance of the model using accuracy, precision, recall, and F1-score metrics.

9. Visualize the results, such as plotting the confusion matrix, using the functions provided in the `shared_func` module.

10. Adjust the model parameters, preprocess the data, or explore different evaluation techniques to improve the model's performance.

## Models

### DTW (Dynamic Time Warping)

The DTW model is based on the Dynamic Time Warping algorithm, which measures the similarity between two time series by warping one series to align with the other. It uses k-nearest neighbors for classification based on the calculated distances.

The `DTW` class provides the following methods:

- `fit(data, labels)`: Fits the DTW classifier with the training dataset and corresponding labels.
- `predict(test_data)`: Predicts the class of the test data using the trained model.
- `compute_distance_matrix(test_data)`: Computes the distance matrix between the test data and the training data.
- `dtw_distance(time_serie1, time_serie2)`: Computes the DTW distance between two time series.
- `distance(x, y)`: Calculates the distance between two points.

###

 1$ Recognizer

The 1$ Recognizer model is an implementation of the $P (one-dollar) gesture recognition algorithm. It represents gestures as sequences of points and compares them using geometric matching techniques. The algorithm uses templates to recognize gestures based on a similarity score.

The `OneDollar` class provides the following methods:

- `fit(train_set, labels)`: Fits the 1$ Recognizer with the training dataset and corresponding labels.
- `predict(test_set)`: Predicts the class of the test set using the trained model.

## Dataset

To train the gesture recognition system, a publicly available gesture dataset, gathered and described in Huang et al. (2019), is used.

- Dataset 01 (Domain 1): This dataset consists of ten users/subjects who were each asked to draw ten different numbers (0-9) in 3D. It contains ten repetitions of each number by ten subjects, resulting in a total of 1000 sequences.

- Dataset 02 (Domain 4): In this dataset, each of the ten subjects was asked to draw ten repetitions of three-dimensional figures, such as pyramids, as described in Huang et al. (2019).

Please note that only Domain 1 and Domain 4 data are used in this implementation.

## Evaluation

The evaluation of the gesture recognition models includes various metrics such as accuracy, precision, recall, and F1-score. The `accuracy_score()`, `precision_score()`, `recall_score()`, and `f1_score()` functions from the `sklearn.metrics` module can be used for evaluation.

The `plot_conf_mat()` function provided in the `shared_func` module can be used to visualize the confusion matrix, which provides insights into the model's performance.

## Contributions

This project was made by Miguel Vicente, Jacopo Palombarini and João Gomes

--
