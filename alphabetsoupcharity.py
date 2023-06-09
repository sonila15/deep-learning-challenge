# -*- coding: utf-8 -*-
"""AlphabetSoupCharity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R9aoZg6uHB91Xo_uKM7L-z12XofWZZJ7

## Preprocessing

# New Section
"""

# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import tensorflow as tf


#  Import and read the charity_data.csv.
import pandas as pd 
application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
application_df.head()

# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.

feature_columns = application_df.drop(["EIN", "NAME"], axis=1)
feature_columns. head()

# Determine the number of unique values in each column.
unique_counts = feature_columns.nunique()
unique_counts

# Look at APPLICATION_TYPE value counts for binning
application_counts = feature_columns['APPLICATION_TYPE'].value_counts()
print(application_counts)

# Choose a cutoff value and create a list of application types to be replaced
# use the variable name `application_types_to_replace`
cutoff = 100
application_types_to_replace = application_counts[application_counts < cutoff].index.tolist()


# Replace in dataframe
for app in application_types_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app,"Other")

# Check to make sure binning was successful
application_df['APPLICATION_TYPE'].value_counts()

# Look at CLASSIFICATION value counts for binning
classification_counts = application_df['CLASSIFICATION'].value_counts()
print(classification_counts)

# You may find it helpful to look at CLASSIFICATION value counts >1
classification_counts = application_df['CLASSIFICATION'].value_counts()
classification_counts_greater_than_1 = classification_counts[classification_counts > 1]
print(classification_counts_greater_than_1)

# Choose a cutoff value and create a list of classifications to be replaced
# use the variable name `classifications_to_replace`
cutoff = 10
classifications_to_replace = classification_counts[classification_counts < cutoff].index.tolist()


# Replace in dataframe
for cls in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls,"Other")
    
# Check to make sure binning was successful
application_df['CLASSIFICATION'].value_counts()

# Convert categorical data to numeric with `pd.get_dummies`
categorical_columns = ['CLASSIFICATION','APPLICATION_TYPE', 'AFFILIATION', 'USE_CASE', 'ORGANIZATION', 'SPECIAL_CONSIDERATIONS', 'INCOME_AMT', 'STATUS']

encoded_data = pd.get_dummies(feature_columns, columns=categorical_columns)

encoded_data. head()

# Split our preprocessed data into our features and target arrays

X = encoded_data.drop('IS_SUCCESSFUL', axis=1)
y = encoded_data['IS_SUCCESSFUL']

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)


# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

"""## Compile, Train and Evaluate the Model"""

# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
#  YOUR CODE GOES HERE

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=80, activation="relu", input_dim=117))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=30, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the target labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)
# Train the model
fit_model = nn.fit(X_train_scaled, y_train_encoded, epochs=100)

# Evaluate the model using the test data
# Convert string target values to float

model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Export our model to HDF5 file
fit_model.model.save("AlphabetSoupCharity.h5")