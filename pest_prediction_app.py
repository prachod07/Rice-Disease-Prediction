#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = r"D:\data analysis\mini\Mini_Re\cleaned_data.csv"

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Define pst_disease_name_encoding dictionary
pst_disease_name_encoding = {1: 'Rice stem borer', 2: 'Paddy Leaf Roller', 3: 'Rice Blue Beetle',
                             4: 'Rice Bug', 5: 'Rice Swarming Caterpillar', 6: 'Rice Blast',
                             7: 'Bacterial Leaf Blight', 8: 'Rice Gall Midge', 9: 'Brown Leaf Spot',
                             10: 'Rice Case Worm', 11: 'Tungro', 12: 'Brown Plant Hopper',
                             13: 'Green Leaf Hopper', 14: 'Whorl Maggot', 15: 'Rice Hispa',
                             16: 'Rice Thrips', 17: 'Bacterial Leaf Streak', 18: 'Sheath Blight',
                             19: 'Sheath Rot', 20: 'False Smut', 21: 'Rice Mealy Bug', 22: 'Udbatta'}

# Use the tuned parameters for the Random Forest Classifier
classifier = RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100)

# Assuming 'data_encoded' is your DataFrame
X = data[['month', 'stageOfCrop_encoded', 'districts_encoded']]
y = data['pstDiseaseName_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the tuned classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=1)

print(f'Tuned Model Accuracy: {accuracy}')
print('Classification Report:\n', classification_rep)

# Backencoding dictionaries
stage_of_crop_encoding = {'Mid tillering': 1, 'Seedling': 2, 'Early tillering': 3, 'Ripening': 4,
                          'Germination and emergence': 5, 'Flowering': 6, 'Milky stage': 7,
                          'Panicle initiation': 8, 'Dough stage': 9, 'Mature': 10, 'Boot': 11}

district_encoding = {'Thiruvananthapuram': 1, 'Kollam': 2, 'Pathanamthitta': 3, 'Alappuzha': 4,
                     'Kottayam': 5, 'Idukki': 6, 'Ernakulam': 7, 'Thrissur': 8,
                     'Palakkad': 9, 'Malappuram': 10, 'Kozhikode': 11, 'Wayanad': 12,
                     'Kannur': 13, 'Kasaragod': 14}

# Streamlit UI
st.title('Pest Prediction App')

# Input fields
month = st.selectbox('Select Month', range(1, 13))
stage_of_crop = st.selectbox('Select Stage of Crop', list(stage_of_crop_encoding.keys()))
district = st.selectbox('Select District', list(district_encoding.keys()))

# Predict button
if st.button('Predict'):
    # Encode the selected values
    stage_of_crop_encoded = stage_of_crop_encoding[stage_of_crop]
    district_encoded = district_encoding[district]

    # Make prediction
    input_data = {'month': month, 'stageOfCrop_encoded': stage_of_crop_encoded, 'districts_encoded': district_encoded}
    input_df = pd.DataFrame([input_data])
    prediction = classifier.predict(input_df)[0]

    # Backencode the prediction
    pest_name = pst_disease_name_encoding[prediction]

    # Display prediction
    st.success(f'Predicted Pest: {pest_name}')


# In[ ]:




