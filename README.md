# Drug-Dosage-Prediction-using-ML
My first repository on ML
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
data = 'drugdos.csv'
df = pd.read_csv(data)

# Asuming df is your DataFrame and 'column_to_remove' is the column you want to remove
column_to_remove = 'review'  # Replace with the actual column name
df = df.drop(column_to_remove, axis=1)


# Define categorical columns
cat_columns = ['drug', 'bp', 'sugar', 'Sideeffects']

# Filter columns that are present in the DataFrame
cat_columns = [col for col in cat_columns if col in df.columns]

# Split the data into training and testing sets
X = df.drop(['dosage', 'condition', 'rating', 'usefulCount', 'temperature'], axis=1)
Y = df['dosage']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a column transformer with OneHotEncoder and SimpleImputer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), cat_columns),
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['number']).columns)
    ],
    remainder='passthrough'
)

# Create a pipeline with the column transformer and the RandomForestRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Train the model
pipeline.fit(X_train, Y_train)

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {
        'age': int(request.form['age']),
        'sugar': request.form['sugar'],
        'bp': request.form['bp'],
        'drug': request.form['drug'],
        'Sideeffects': request.form['Sideeffects']
    }

    user_data_df = pd.DataFrame([user_data])
    dosage_prediction = pipeline.predict(user_data_df)[0]

    return render_template('result.html', result=f'Predicted Dosage: {dosage_prediction}')

if __name__ == '__main__':
    app.run(debug=True)

# home.html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Dosage Prediction</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>ML Dosage Prediction</h1>
        <form action="/predict" method="post">
            <label for="age">Age:</label>
            <input type="number" name="age" required><br>
            <label for="sugar">Sugar Level (normal/high):</label>
            <select name="sugar" class="dropdown" required>
                <option value="abnormal">abnormal</option>
                <option value="normal">normal</option>
            </select><br>
            <label for="bp">Blood Pressure (normal/abnormal):</label>
            <select name="bp" class="dropdown" required>
                <option value="abnormal">abnormal</option>
                <option value="normal">normal</option>
            </select><br>
            <label for="drug">Drug:</label>
            <select name="drug" class="dropdown" required>
                <!-- ... options ... -->
                <option value="Mirtazapine">Mirtazapine</option>
        <option value="Mesalamine">Mesalamine</option>
        <option value="Bactrim">Bactrim</option>
        <option value="Contrave">Contrave</option>
        <option value="LEVORA">LEVORA</option>
        <option value="Miconazole">Miconazole</option>
        <option value="Nuvigil">Nuvigil</option>
        <option value="Ciprofloxacin">Ciprofloxacin</option>
        <option value="Trazodone">Trazodone</option>
        <option value="Aripiprazole">Aripiprazole</option>
        <option value="Oxybutynin">Oxybutynin</option>
        <option value="Clonazepam">Clonazepam</option>
        <option value="Sodium oxybate">Sodium oxybate</option>
        <option value="Lamotrigine">Lamotrigine</option>
        <option value="Blisovi Fe 1 / 20">Blisovi Fe 1 / 20</option>
        <option value="Ivermectin">Ivermectin</option>
        <option value="Suprep Bowel Prep Kit">Suprep Bowel Prep Kit</option>
        <option value="Movantik">Movantik</option>
        <option value="Actos">Actos</option>
        <option value="Duloxetine">Duloxetine</option>
        <option value="NuvaRing">NuvaRing</option>
        <option value="Escitalopram">Escitalopram</option>
        <option value="Campral">Campral</option>
        <option value="Gabapentin">Gabapentin</option>
        <option value="Levonorgestrel">Levonorgestrel</option>
        <option value="Aubra">Aubra</option>
        <option value="Ethinyl estradiol / etonogestrel">Ethinyl estradiol / etonogestrel</option>
        <option value="Microgestin Fe 1.5 / 30">Microgestin Fe 1.5 / 30</option>
        <option value="Wellbutrin">Wellbutrin</option>
        <option value="Etonogestrel">Etonogestrel</option>
        <option value="Nitrofurantoin">Nitrofurantoin</option>
        <option value="Beyaz">Beyaz</option>
        <option value="Lo Loestrin Fe">Lo Loestrin Fe</option>
        <option value="Ethinyl estradiol / norgestimate">Ethinyl estradiol / norgestimate</option>
        <option value="Guaifenesin / pseudoephedrine">Guaifenesin / pseudoephedrine</option>
        <option value="Glyburide">Glyburide</option>
        <option value="Phentermine / topiramate">Phentermine / topiramate</option>
            </select><br>
            <label for="Sideeffects">Side-effects:</label>
            <select name="Sideeffects" class="dropdown" required>
                <!-- ... options ... -->
                 <option value="dizziness">dizziness</option>
        <option value="headache">headache</option>
        <option value="vomting">vomting</option>
        <option value="vomting and headache">vomting and headache</option>
        <option value="headache and dizziness">headache and dizziness</option>
        <option value="weight gain">weight gain</option>
        <option value="heatburn">heatburn</option>
            </select><br>
            <button type="submit">Predict Dosage</button>
        </form>
    </div>
</body>
</html>

# result.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Dosage Prediction Result</title>
    <link rel="stylesheet" href="/static/style.css">

</head>
<body>
   <div class="container">
    <h1>ML Dosage Prediction Result</h1>
    <p>{{ result }}</p>
    <a href="/" id="link">Go back to home</a>
       </div>
</body>
</html>

# style.css
body {
    margin: 0;
    padding: 0;
    background-image: url('/static/images/3.avif');
    background-size: cover;
    background-position: center;
    background-color: #f0f0f0; /* Fallback color if the image is not loaded */
    font-family: 'Arial', sans-serif;
}

.container {
    background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white background for the container */
    padding: 40px;
    margin: 40px;
    border-radius: 10px;
    height:50%;
    width:40%;
    margin-left:400px;
}

h1 {
    color: #3498db;
    text-align: center;
}

form {
    max-width: 400px;
    margin: 0 auto;
}

label {
    display: block;
    margin-bottom: 5px;
    color: #3498db;
    font-weight: bold;
}

input, select {
    width: 100%;
    padding: 8px;
    margin-bottom: 10px;
    color: green;
}

button {
    background-color: #3498db;
    color: #fff;
    padding: 10px;
    border: none;
    cursor: pointer;
    text-align: center;
    width: 100%;
}

button:hover {
    background-color: #2980b9;
}

.dropdown {
    width: 100%;
    padding: 8px;
    margin-bottom: 10px;
    color: green;
    font-weight: bold;
}

p {
    text-align: center;
    font-size: large;
    color: black;
    font-weight: bold;
}
#link{
  align:center;
}


