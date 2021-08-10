# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

The dataset is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The y column indicates if a customer subscribed to a fixed term deposit, which is later identified as the target column for predictions in this tutorial.

- In this project, we have used HyperDrive to tune and find optimum parameters for scikit-learn Logistic Regression model.
- We have also used AutoML to build a large set of models and pick the best performing model (VotingEnsemble) with accuracy of 0.9179.

## Scikit-learn Pipeline

1. Setup Training Script
    - Data is imported using TabularDatasetFactory
    - Data Pre-processing steps like cleaning of data (handling NULL/None/NaN values), preprocessing of date variables is carried out. 
    - Splitting of data into train and test sets has been done.
    - Used Sklearn model with logistic regression as objective for classification

2. Create SKLearn Estimator for training the model selected (logistic regression) by passing the training script and later the estimator is passed to the hyperdrive configuration.

3. Configuration of Hyperdrive
    - Selection of parameter sampler
    - Selection of primary metric
    - Selection of early termination policy
    - Selection of estimator (SKLearn)
    - Allocation of resources

4. Save the trained optimized model

We have used SKLearn Estimator with logistic regression for binary classification. Hyperdrive is used to find optimum set of parameters in th given space.

You can see the Hyperdrive models in the below image :

![Hyperdrive](/images/hyperdrive.png)


You can see the Hyperdrive models metrics in the below image :

![Hyperdrive Accuracy](/images/hyperdrive-accuracy.png)

### Parameter Sampler

- The parameter sampler I have used was RandomParameterSampling because it supports both discrete and continuous hyperparameters. 
- It supports early termination of low-performance models and supports early stopping policies. 
- In random sampling , the hyperparameter (C : smaller values specify stronger regularization, max_iter : maximum number of iterations taken for the solvers to converge) values are randomly selected from the given search space.

### Early Stopping Policy

- The early stopping policy I have used was BanditPolicy because it is based on slack factor and evaluation interval. 
- BanditPolicy terminates runs where the primary metric is not within the specified slack factor compared to the best performing run.

## AutoML

- AutoML is used to run experiments to generate large set of models and compare their performance and choose one.
- Data is imported using TabularDatasetFactory
- Data Pre-processing steps like cleaning of data is carried out. 
- Splitting of data into train and test sets has been done.
- Provided AutoML config with required parameters.
- AutoML performace has been observed using RunDetails and best performing model is saved.

### AutoML Parameters used

I have used several parameters for AutoML config :

- experiment_timeout_minutes - This parameter ensures the AutoML experiemnts don't run for longer durations.
- task - Kind of ML task to be performed - e.g. classification/regression.
- primary_metric - Name of the primary metric to be used. e.g. "accuracy".
- training_data - pandas dataframe / numpy array of training data.
- label_column_name - name of target variable.
- n_cross_validations - No. of cross validation folds to be used in building AutoML models.
- compute_target - Azure ML Compute instance where the AutoML needs to be run.
- enable_early_stopping - This boolean parameter ensures that AutoML training doesn't take too much time and stops once the primary_metric reaches a saturation point above which the accuracy would not increase. 
- enable_onnx_compatible_models - This parameter boolean makes sure that trained model can be exported in Onnx compatible format.

More information regarding AutoML can be found here : (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml)

![AutoML Comparision](/images/automl-comparision.png)

Below image shows metrics for AutoML model :
![AutoML Accuracy](/images/automl-metrics.png)
![AutoML Accuracy](/images/confusion-matrix.png)

## Pipeline comparison

- Both the approaches - Sklearn + Hyperdrive and AutoML follow similar data processing steps and the difference lies in their configuration details. 
- In the Hyperdrive approach, the model is fixed and we use hyperdrive tool to find optimal hyperparametets in th given space
- In the AutoML approach, different models are automatic generated with their own optimal hyperparameter values and the best model is selected. 

- In the below image, we see that the hyperdrive approach took overall 2m 3s and the best model had an accuracy of ~0.9129 and the automl approach took overall 43m 29s and the best model had an acccuracy of ~0.91794.

![Hyperdrive Accuracy](/images/combined.png)

- It is quite evident from the comparision that AutoML was able to generate slightly better model in terms of accuracy metric.
- AutoML takes significantly longer time as it generates new ML Models with optimum parameter config., whereas Hyperdrive only optimizes parameter space for single model.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

- Explore more models like XGBoost, Catboost to use them with Hyperdrive for finding best parameters. This will give scope for is to try new models which can give better accuracy.
- Different sampling methods can be used with Hyperdrive instead of Random Sampling for parameter selection. This might give us different set of results.
- Do more extensive exploratory analysis on the bank marketing dataset so that we can do feature engineering. New features can be created and added to the training features which can improve the accuracy further.
- Also, there is class imbalance in the dataset, which has slight affect on accuracy. If we can address the issue with techniques like over-sampling, under-sampling or SMOTE, the accuracy can improve further.
- For AutoML, we can experiment with different experiment_timeout_minutes, n_cross_validations and it might increase the performance if the model as more models can be created.

## Proof of cluster clean up

**Image of cluster marked for deletion**
![AutoML Accuracy](/images/cluster-deletion.png)