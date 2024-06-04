import streamlit as st

import pandas as pd





import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load your dataset
df = pd.read_csv('/Users/williamlu/Desktop/INTF/recs2015_public_v4.csv')

# Select relevant columns
relevant_columns = ['KWH', 'TOTHSQFT', 'EQUIPAGE', 'TOTROOMS', 'NHSLDMEM', 'TOTCSQFT', 'HDD65','TYPEHUQ','NUMBERAC','ELWARM','ELWATER','ELFOOD'] #Elwarm good, ELFOOD good, ELWATER
df = df[relevant_columns]

# Check for any missing values
print(df.isnull().sum())

# Display the first few rows to confirm correct data loading
print(df.head())



# Handling missing values
df.dropna(inplace=True)

# Ensuring all data types are correct for modeling
df['EQUIPAGE'] = df['EQUIPAGE'].astype(float)
df['TOTHSQFT'] = df['TOTHSQFT'].astype(float)
df['TOTROOMS'] = df['TOTROOMS'].astype(float)
df['NHSLDMEM'] = df['NHSLDMEM'].astype(float)
df['TOTCSQFT'] = df['TOTCSQFT'].astype(float)
df['HDD65'] = df['HDD65'].astype(float)
df['TYPEHUQ'] = df['TYPEHUQ'].astype(float)
df['NUMBERAC '] = df['NUMBERAC'].astype(float)


# Adding a small constant to avoid log(0) issues later
df['TOTHSQFT'] = df['TOTHSQFT'].replace(0, np.nan)
df['TOTROOMS'] = df['TOTROOMS'].replace(0, np.nan)
df['NHSLDMEM'] = df['NHSLDMEM'].replace(0, np.nan)
df['EQUIPAGE'] = df['EQUIPAGE'].replace(0, np.nan)
df['TOTCSQFT'] = df['TOTCSQFT'].replace(0, np.nan)
df['HDD65'] = df['HDD65'].replace(0, np.nan)
df['NUMBERAC'] = df['NUMBERAC'].replace(-2, np.nan)




# Drop rows with NaN values after replacements
df.dropna(inplace=True)

# Display a summary of the cleaned data
print(df.describe())


# Applying logarithmic transformations to stabilize variance and reduce skewness
df['KWH_log'] = np.log1p(df['KWH'])
df['TOTHSQFT_log'] = np.log1p(df['TOTHSQFT'])
df['TOTROOMS_log'] = np.log1p(df['TOTROOMS'])
df['NHSLDMEM_log'] = np.log1p(df['NHSLDMEM'])
df['EQUIPAGE_log'] = np.log1p(df['EQUIPAGE'])
df['TOTCSQFT_log'] = np.log1p(df['TOTCSQFT'])
df['HDD65_log'] = np.log1p(df['HDD65'])
df['TYPEHUQ_log'] = np.log1p(df['TYPEHUQ'])
df['NUMBERAC_log'] = np.log1p(df['NUMBERAC'])
df['ELWARM_log'] = np.log1p(df['ELWARM'])
df['ELFOOD_log'] = np.log1p(df['ELFOOD'])
df['ELWATER_log'] = np.log1p(df['ELWATER'])

# Cell for removing outliers based on IQR

def remove_outliers(df, column):
    # Calculate the IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the acceptable range (typically 1.5*IQR below Q1 and above Q3)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data to remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# List of log-transformed variables to process
log_vars = ['KWH_log', 'TOTHSQFT_log', 'TOTROOMS_log', 'NHSLDMEM_log', 'EQUIPAGE_log', 
            'TOTCSQFT_log', 'HDD65_log', 'TYPEHUQ_log', 'NUMBERAC_log', 'ELWARM_log', 
            'ELFOOD_log', 'ELWATER_log']

# Apply the outlier removal for each log-transformed variable
for var in log_vars:
    df = remove_outliers(df, var)

# Display the first few rows of the cleaned dataset
print(df.head())


# Import the statsmodels library for regression analysis
import statsmodels.api as sm

# Prepare the independent variables (adding a constant term for the intercept)
X = sm.add_constant(df[['TOTCSQFT_log', 'TOTROOMS_log','NHSLDMEM_log','HDD65_log','TYPEHUQ_log','NUMBERAC_log','ELFOOD_log','ELWARM_log','ELWATER_log']])

# Define the dependent variable
y = df['KWH_log']

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(model.summary())



# Cell 6: Polynomial Regression using StatsModels

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# Prepare the independent variables with polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['TOTCSQFT_log', 'TOTROOMS_log', 'NHSLDMEM_log', 'HDD65_log', 'TYPEHUQ_log', 'NUMBERAC_log', 'ELFOOD_log', 'ELWARM_log', 'ELWATER_log']])

# Add a constant term for the intercept in the model
X_poly = sm.add_constant(X_poly)

# Fit the polynomial regression model
model_poly = sm.OLS(y, X_poly).fit()

# Print the summary of the regression results
print(model_poly.summary())





def Ppredict_kwh(TOTCSQFT, TOTROOMS, NHSLDMEM, HDD65, TYPEHUQ, NUMBERAC, ELFOOD, ELWATER, ELWARM, model_poly, poly):
    import numpy as np
    import statsmodels.api as sm

    # Convert input data to log scale, adjusting for special cases as per your earlier transformation steps
    inputs = np.array([
        np.log1p(TOTCSQFT),
        np.log1p(TOTROOMS),
        np.log1p(NHSLDMEM),
        np.log1p(HDD65),
        np.log1p(TYPEHUQ),
        np.log1p(max(NUMBERAC, 0)),  # handling special case for NUMBERAC
        np.log1p(ELFOOD),
        np.log1p(ELWARM),
        np.log1p(ELWATER)
    ]).reshape(1, -1)
    
    # Apply polynomial transformation using the same PolynomialFeatures instance used in training
    inputs_poly = poly.transform(inputs)

    # Ensure the transformed inputs are correctly shaped and named if necessary
    if hasattr(poly, 'get_feature_names_out'):
        input_df = pd.DataFrame(inputs_poly, columns=poly.get_feature_names_out())
    else:
        input_df = pd.DataFrame(inputs_poly, columns=poly.get_feature_names())

    # Add a constant term for the intercept
    input_df = sm.add_constant(input_df, has_constant='add')

    # Predict using the polynomial model
    prediction_log = model_poly.predict(input_df)
    
    # Transform back from log scale
    prediction = np.expm1(prediction_log)
    
    return prediction

# Example of using the function
# Make sure to pass the 'model_poly' and 'poly' that were used in training
print(Ppredict_kwh(2500, 3, 3, 3000, 1, 3, 1, 1, 1, model_poly, poly))



def predict_kwh(TOTCSQFT, TOTROOMS, NHSLDMEM, HDD65, TYPEHUQ, NUMBERAC, ELFOOD, ELWARM, ELWATER, model):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # Log transform the input variables
    inputs = {
        'TOTCSQFT_log': np.log1p(TOTCSQFT),
        'TOTROOMS_log': np.log1p(TOTROOMS),
        'NHSLDMEM_log': np.log1p(NHSLDMEM),
        'HDD65_log': np.log1p(HDD65),
        'TYPEHUQ_log': np.log1p(TYPEHUQ),
        'NUMBERAC_log': np.log1p(NUMBERAC),
        'ELFOOD_log': np.log1p(ELFOOD),
        'ELWARM_log': np.log1p(ELWARM),
        'ELWATER_log': np.log1p(ELWATER)
    }

    # Convert inputs into a DataFrame
    input_df = pd.DataFrame([inputs])

    # Ensure the DataFrame includes a constant term for the intercept
    input_df_with_const = sm.add_constant(input_df, has_constant='add')

    # Predict using the model
    prediction_log = model.predict(input_df_with_const)

    # Convert log prediction back to original scale
    prediction = np.expm1(prediction_log.iloc[0])

    return prediction

# Example usage of the function
predicted_kwh = predict_kwh(2500, 3, 3, 3000, 1, 3, 1, 1, 1, model)
print(f"Predicted KWh: {predicted_kwh}")


st.write("This is a message in Streamlit!")



# Define HDD65 values for regions
hdd65_mapping = {
    "Northeast": 5000,
    "Midwest": 5500,
    "South": 2000,
    "West": 3000
}

# Define yes/no options for electricity usage
yes_no_options = {
    "Yes": 1,
    "No": 0
}



# Page title
st.title("Energy Consumption Prediction Model by William")

# User Inputs with Sliders
TOTCSQFT = st.slider('Total House Square Footage', min_value=500, max_value=5000, value=1500, step=10)
TOTROOMS = st.slider('Total Number of Rooms', min_value=1, max_value=20, value=5, step=1)
NHSLDMEM = st.slider('Number of Household Members', min_value=1, max_value=10, value=3, step=1)


# Region to HDD65 Mapping
region_to_hdd65 = {
    "Northeast": 6000,
    "Midwest": 7000,
    "South": 2000,
    "West": 4000
}
region = st.selectbox('Select Your Region', list(region_to_hdd65.keys()))
HDD65 = region_to_hdd65[region]

# Type of Housing Unit
type_housing = {
    "Mobile home": 1,
    "Single-family detached house": 2,
    "Single-family attached house": 3,
    "Apartment in a building with 2 to 4 units": 4,
    "Apartment in a building with 5 or more units": 5
}
TYPEHUQ = st.selectbox('Type of Housing Unit', list(type_housing.keys()), format_func=lambda x: x)
TYPEHUQ = type_housing[TYPEHUQ]

# Electricity Usage
ELFOOD = st.selectbox('Electricity for Food', ['No', 'Yes'], index=0)
ELFOOD = 1 if ELFOOD == 'Yes' else 0
ELWARM = st.selectbox('Electricity for Warming', ['No', 'Yes'], index=0)
ELWARM = 1 if ELWARM == 'Yes' else 0
ELWATER = st.selectbox('Electricity for Water', ['No', 'Yes'], index=0)
ELWATER = 1 if ELWATER == 'Yes' else 0

NUMBERAC = st.number_input('Number of Air Conditioners', min_value=0, max_value=4)




# Buttons to perform prediction
if st.button('Predict with Linear Model'):
    predicted_linear = predict_kwh(TOTCSQFT, TOTROOMS, NHSLDMEM, HDD65, TYPEHUQ, 2, ELFOOD, ELWARM, ELWATER, model)
    st.write(f'Predicted KWh (Linear): {predicted_linear}')

if st.button('Predict with Polynomial Model'):
    predicted_poly = Ppredict_kwh(TOTCSQFT, TOTROOMS, NHSLDMEM, HDD65, TYPEHUQ, 2, ELFOOD, ELWARM, ELWATER, model_poly, poly)
    st.write(f'Predicted KWh (Polynomial): {predicted_poly}')


