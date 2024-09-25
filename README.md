# Student Loan Risk with Deep Learning

## Overview

This project aims to predict student loan repayment success using a deep learning model. We leverage a neural network to classify whether an individual is likely to repay their loan successfully based on various features provided in the dataset. The deep learning model will be trained, tested, and used for making predictions on unseen data.

## Requirements

To complete this project, you'll need the following Python libraries:

* pandas
* tensorflow
* scikit-learn
* pathlib

These libraries help in data handling, model building, scaling, and evaluation.

## Instructions

**Step 1: Prepare the Data for the Neural Network Model**

Read and Review the Data: Begin by loading the student-loans.csv file into a Pandas DataFrame. Take a moment to analyze the data to identify which columns should be used as features and which will serve as the target variable for the model.

Define Features and Target: The credit_ranking column will be the target variable, representing whether a loan is likely to be repaid. The other columns will be used as features to predict this outcome.

Split the Data: After defining the features and target, the dataset will be divided into training and testing sets. This split ensures that the model can be evaluated on unseen data after training.

Scale the Features: Feature scaling is necessary to standardize the dataset. Using a scaler will help the model perform optimally by normalizing the data values.

**Step 2: Build, Compile, and Evaluate the Neural Network Model**

Model Creation: A deep neural network will be created using Tensorflow’s Keras. This includes defining the number of layers, neurons in each layer, and activation functions. The structure of the network will be tuned based on the number of input features.

Model Compilation: The neural network will be compiled using a binary classification setup. The loss function used will be binary_crossentropy, the optimizer will be adam, and the evaluation metric will be accuracy.

Train and Fit the Model: The model will be trained using the scaled training dataset and evaluated using validation data to monitor performance over multiple epochs.

Model Evaluation: Once the model has been trained, it will be evaluated using the test dataset. The evaluation will return the accuracy and loss, which indicate how well the model performs in predicting loan repayment success.

Save the Model: After the evaluation, the model will be saved in a Keras file format for future use.

**Step 3: Predict Loan Repayment Success**

Reload the Saved Model: To make future predictions, the saved neural network model will be reloaded from the file.

Make Predictions: Using the testing data, predictions will be generated. The results will be stored in a DataFrame for further analysis.

Classification Report: A classification report will be displayed to summarize the model’s performance, including precision, recall, and F1-score metrics based on the predicted and actual test data.

## Conclusion

This project demonstrates how a deep learning neural network can be used to predict student loan repayment success. By following the steps outlined above, you will build and evaluate a model that can be used to make future predictions, providing valuable insights into loan repayment risk.







