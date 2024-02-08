# app.py

from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Sample data loading and model training (replace with your actual data and model training)
file_path = r"D:\Projects\Rice Disease\webapp\cleaned_data.csv"
data = pd.read_csv(file_path)

# Assuming 'data_encoded' is your DataFrame
X = data[['month', 'stageOfCrop_encoded', 'districts_encoded']]
y = data['pstDiseaseName_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use the tuned parameters for the Random Forest Classifier
classifier = RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100)
classifier.fit(X_train, y_train)

# Define encoding dictionaries
stage_of_crop_encoding = {'Mid tillering': 1, 'Seedling': 2, 'Early tillering': 3, 'Ripening': 4,
                          'Germination and emergence': 5, 'Flowering': 6, 'Milky stage': 7,
                          'Panicle initiation': 8, 'Dough stage': 9, 'Mature': 10, 'Boot': 11}

district_encoding = {'Thiruvananthapuram': 1, 'Kollam': 2, 'Pathanamthitta': 3, 'Alappuzha': 4,
                     'Kottayam': 5, 'Idukki': 6, 'Ernakulam': 7, 'Thrissur': 8,
                     'Palakkad': 9, 'Malappuram': 10, 'Kozhikode': 11, 'Wayanad': 12,
                     'Kannur': 13, 'Kasaragod': 14}

# Mapping of predicted pests to image filenames
pest_image_mapping = {
    'Rice Bug': 'Rice Bug.jpg', 'Rice Case Worm': 'Rice Case Worm.jpg',
    'Paddy Leaf Roller': 'Paddy Leaf Roller.jpg',
    'Rice Blast': 'Rice Blast.jpg'  # Add other pests and their image filenames
}

# Information about diseases and their treatments
disease_info = {
    'Rice Bug': ' A multifaceted approach encompassing cultural, biological, and mechanical practices. This includes adopting measures such as crop rotation, encouraging natural predators, and employing manual removal or traps, while judicious use of insecticides may be considered for severe infestations.',
    'Paddy Leaf Roller': ' Specific pesticides such as Bacillus thuringiensis (Bt), neem-based formulations, or chemical insecticides like cypermethrin. Cultural practices like removing affected leaves and maintaining proper field hygiene also contribute to effective management. Consult with agricultural experts or local extension services for detailed and updated recommendations based on your specific conditions.',
    'Rice Blast': ' The application of fungicides such as tricyclazole, propiconazole, or isoprothiolane during the critical stages of crop development. Cultural practices like maintaining appropriate plant spacing, using disease-resistant varieties, and practicing proper water management can also help in preventing and managing Rice Blast. For accurate and updated recommendations its advisable to consult with agricultural experts or local extension services based on your specific geographical and climatic conditions.',
    # Add information for other diseases
}

@app.route('/')
def home():
    return render_template('index.html', stage_of_crop_encoding=stage_of_crop_encoding, district_encoding=district_encoding)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        month = int(request.form['month'])
        stage_of_crop = request.form['stage_of_crop']
        district = request.form['district']

        # Encode the selected values
        stage_of_crop_encoded = stage_of_crop_encoding.get(stage_of_crop, 1)
        district_encoded = district_encoding.get(district, 1)

        # Make prediction
        input_data = {'month': month, 'stageOfCrop_encoded': stage_of_crop_encoded, 'districts_encoded': district_encoded}
        input_df = pd.DataFrame([input_data])
        prediction = classifier.predict(input_df)[0]

        # Backencode the prediction
        pst_disease_name_encoding = {1: 'Rice stem borer', 2: 'Paddy Leaf Roller', 3: 'Rice Blue Beetle',
                                     4: 'Rice Bug', 5: 'Rice Swarming Caterpillar', 6: 'Rice Blast',
                                     7: 'Bacterial Leaf Blight', 8: 'Rice Gall Midge', 9: 'Brown Leaf Spot',
                                     10: 'Rice Case Worm', 11: 'Tungro', 12: 'Brown Plant Hopper',
                                     13: 'Green Leaf Hopper', 14: 'Whorl Maggot', 15: 'Rice Hispa',
                                     16: 'Rice Thrips', 17: 'Bacterial Leaf Streak', 18: 'Sheath Blight',
                                     19: 'Sheath Rot', 20: 'False Smut', 21: 'Rice Mealy Bug', 22: 'Udbatta'}

        pest_name = pst_disease_name_encoding.get(prediction, 'Unknown')

        # Get the image filename based on the predicted pest
        pest_image_filename = pest_image_mapping.get(pest_name, 'default_image.jpg')

        return render_template('index.html', stage_of_crop_encoding=stage_of_crop_encoding, district_encoding=district_encoding, pest_name=pest_name, pest_image_filename=pest_image_filename, disease_info=disease_info)

if __name__ == '__main__':
    app.run(debug=True)
