# deep-learning-challenge

# Report on the Neural Network Model

## Overview of the Analysis
The purpose of this analysis is to develop a deep learning model using a neural network for Alphabet Soup, a venture capital firm. The goal is to create a model that can predict the success of funding applicants based on various features provided in the dataset. Accurately identifying successful funding ventures will enable Alphabet Soup to optimize their funding decisions and allocate resources more effectively.

## Results
### Data Preprocessing
Target Variable: The "IS_SUCCESSFUL" column is the target variable, indicating whether a funding venture was successful (1) or not (0).
Feature Variables: The model uses several feature variables, including "APPLICATION_TYPE," "AFFILIATION," "CLASSIFICATION," "USE_CASE," "ORGANIZATION," "STATUS," "INCOME_AMT," and "SPECIAL_CONSIDERATIONS."
Variables Removed: The "EIN" and "NAME" variables were removed from the input data as they do not contribute to the prediction.
In the second model, I reduced the number of rows in the file to address the crashing issue that occurred after some time. The target and feature variables were almost the same, except for adding the "NAME" column as a feature variable and dropping the "SPECIAL_CONSIDERATIONS" column.

### Compiling, Training, and Evaluating the Model
### Neurons, Layers, and Activation Functions: 
The neural network model consists of three hidden layers. The first hidden layer has 80 neurons, the second has 30 neurons, and the third has 10 neurons. The rectified linear unit (ReLU) is used as the activation function for all hidden layers. The output layer has a single neuron with a sigmoid activation function, predicting the binary classification of success or failure.

### Target Model Performance: 
The target model performance is to achieve an accuracy of at least 75% on the test dataset.
- In the first model, the accuracy and loss results were:

Accuracy: 0.7270
Loss: 0.5617
- In the second model, the accuracy and loss results were:

Accuracy: 0.7575
Loss: 0.4905

### Steps to Increase Model Performance
Several steps were taken to increase the model's performance:
- Added the "NAME" column as a feature.
- Dropped the "SPECIAL_CONSIDERATIONS" column.
- Adjusted the number of epochs to 20 due to limited storage capacity, which achieved an accuracy of 75.5%.
- Reduced the size of the main file by reducing the number of rows.
- Changed a cuoff values of aplication types.

### Data Preprocessing
The categorical variables were encoded using one-hot encoding to convert them into a numerical representation suitable for the model.

### Feature Selection
Correlation analysis and feature importance techniques were applied to identify and select the most relevant features for the model.

### Model Optimization
Hyperparameter tuning was performed by adjusting parameters such as the number of neurons, layers, and activation functions to find the optimal configuration.

### Model Ensemble
Ensemble techniques, such as bagging or boosting, could be explored to combine multiple models for improved performance.

### Summary
The deep learning model developed using a neural network shows promising results in predicting the success of funding ventures for Alphabet Soup. By utilizing various features and applying data preprocessing techniques, the model achieves a satisfactory level of accuracy. However, there is still room for improvement.

### Recommendation:
For this classification problem, an alternative model that could be considered is the Random Forest Classifier. Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. It can handle categorical variables without the need for one-hot encoding and can handle high-dimensional data well. Additionally, Random Forest can provide feature importances, which can help in understanding the significance of each feature in the prediction task.
