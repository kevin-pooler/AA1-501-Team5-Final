# Check requirments.txt for neccessary libraries we need to install

import gdown  # Allows us to use a Google Drive hosted data file
import pandas as pd

# Google Drive file ID
file_id = '1-CvtEzDoTFsJ5cAnafBS608prTUAl1Vf'
url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Destination file path
output = 'age_gender.csv'

# Download the file
gdown.download(url, output, quiet=False)

# Load the data
df = pd.read_csv(output)

# Display basic information from dataset
print(df.info())
print(df.head())
print(df.describe())
print(df.shape)

#------------------- PREPROCESSING -------------------

# Extract and preprocess pixel data
pixels = df['pixels'].str.split(' ', expand=True).astype(int)

# Convert the 'pixels' column to arrays
pixels = pixels.values

# Reshape arrays to 48x48 images
images = pixels.reshape(-1, 48, 48, 1)  # Adding channel dimension

# Normalize pixel values to the range [0, 1]
images = images / 255.0

# Extract target variables (age, gender, ethnicity)
ages = df['age'].values
genders = df['gender'].values
ethnicities = df['ethnicity'].values

# Handle data imbalances if needed
# ...

# Split data into training and testing sets for age, gender, and ethnicity
# Example: Splitting data for gender prediction

#------------------- MODELS -------------------

# Define and train the Feed Forward Neural Network (FFN)
# ...

# Define and train the Convolutional Neural Network (CNN)
# ...

# Define and train the Vision Transformer (ViT)
# ...

#------------------- EVALUATION -------------------

# Evaluate each model on the test set
# For each model (FFN, CNN, ViT), predict on X_test and calculate performance metrics
# ...

# Compare the performance of the three models
# Summarize the performance metrics for FFN, CNN, and ViT
# Discuss the trade-offs between model complexity, accuracy, and computational efficiency
# ...

#------------------- SAVING MODELS -------------------

# Save the models
# Save each trained model for future use
# ...

#------------------- DOCUMENTATION -------------------

# Document the findings and insights
# Include visualizations of model performance and comparisons
# Add references and documentation for the project
# ...