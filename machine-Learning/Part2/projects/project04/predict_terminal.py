import pandas as pd
import joblib

# Load the trained model
model = joblib.load("final_model.pkl")

# Define the exact expected columns
expected_columns = [
    'age', 'experience',
    'css', 'express', 'html', 'java', 'js', 'node', 'python', 'react', 'spring',
    'company_Microsoft', 'company_Startupx', 'company_Technova',
    'designation_Data Scientist', 'designation_Full Stack Developer',
    'gender_male'
]

# Step 1: Get user input
age = int(input("Enter age: "))
experience = int(input("Enter years of experience: "))
gender = input("Enter gender (Male/Female): ").strip().lower()
company = input("Enter company (Microsoft, Startupx, Technova, Google): ").strip().lower()
designation = input("Enter designation (Full Stack Developer, Data Scientist, Backend Developer): ").strip()
skills_input = input("Enter skills (comma-separated like: html, css, js): ").strip().lower()

# Step 2: Build input dictionary
input_data = {col: 0 for col in expected_columns}
input_data['age'] = age
input_data['experience'] = experience

# Skills: mark as 1 if included
skills = [s.strip() for s in skills_input.split(',')]
for skill in skills:
    if skill in input_data:
        input_data[skill] = 1

# Company
company_col = f'company_{company.capitalize()}'
if company_col in input_data:
    input_data[company_col] = 1

# Designation
designation_col = f'designation_{designation}'
if designation_col in input_data:
    input_data[designation_col] = 1

# Gender
if gender == 'male':
    input_data['gender_male'] = 1  # if not male, leave as 0 (meaning female)

# Step 3: Create DataFrame in the exact column order
input_df = pd.DataFrame([input_data])[expected_columns]

# Step 4: Predict
predicted_salary = model.predict(input_df)[0]

# Output
print(f"\nPredicted Salary: ${predicted_salary:.2f}")
