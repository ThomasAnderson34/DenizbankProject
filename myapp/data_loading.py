import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from flask import Flask, request, jsonify
import joblib
import requests



df = pd.read_csv("dataset.csv", sep=";")  # df is DataFrame
print(df.shape)  # to see the dimension
print(df.head())  # Display the first few rows of the DataFrame
print(df.info())  # Display information about the DataFrame
#########################

# Put attributes into different lists. So it can be seen better when analyzing the data

num_columns = ["account_amount_added_12_24m", "account_days_in_dc_12_24m", "account_days_in_rem_12_24m",
               "account_days_in_term_12_24m",
               "account_incoming_debt_vs_paid_0_24m", "age", "avg_payment_span_0_12m", "avg_payment_span_0_3m",
               "max_paid_inv_0_12m", "max_paid_inv_0_24m",
               "num_active_div_by_paid_inv_0_12m", "num_active_inv", "num_arch_dc_0_12m", "num_arch_dc_12_24m",
               "num_arch_ok_0_12m", "num_arch_ok_12_24m",
               "num_arch_rem_0_12m", "num_arch_written_off_0_12m", "num_arch_written_off_12_24m", "num_unpaid_bills",
               "recovery_debt", "sum_capital_paid_account_0_12m",
               "sum_capital_paid_account_12_24m", "sum_paid_inv_0_12m", "time_hours"]

cat_columns = ["default", "account_status", "account_worst_status_0_3m", "account_worst_status_12_24m",
               "account_worst_status_3_6m", "account_worst_status_6_12m",
               "merchant_category", "merchant_group", "name_in_email", "status_last_archived_0_24m",
               "status_2nd_last_archived_0_24m", "status_3rd_last_archived_0_24m",
               "status_max_archived_0_6_months", "status_max_archived_0_12_months", "status_max_archived_0_24_months",
               "worst_status_active_inv"]

bool_columns = ["has_paid"]

# See how many missing values are there in each attribute
missing_values = df.isna().sum()
print("\nMISSING VALUES\n", missing_values)

# Calculate the percentage of missing values in each column
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Sort the columns by their missing percentage in descending order
missing_percentage = missing_percentage.sort_values(ascending=False)

# Print the title
print("\nMISSING PERCENTAGE\n")

# Print the missing percentage for each column
print(missing_percentage)

# Plot the missing percentage and see it in visual
plt.figure(figsize=(12, 8))
sns.barplot(x=missing_percentage.index, y=missing_percentage.values)
plt.xticks(rotation=90)
plt.ylabel("Missing Percentage")
plt.xlabel("Columns")
plt.title("Percentage of Missing Values in Each Column")

# Highlight columns with missing values above the threshold
threshold = 49
above_threshold = missing_percentage[missing_percentage > threshold]
for i, v in enumerate(above_threshold):
    plt.text(i, v + 1, f"{v:.2f}%", ha="center")

plt.show()

# Use value_counts and see how many different variables are there in each column
for col in cat_columns:
    print("Column Name: ", '\033[34m', col, '\033[0m')
    print(df[col].value_counts(), "\n")

# Separate NA default variable from the data and put into a new object named predict_data
# I will use this object in the prediction step
predict_data = df[df['default'].isnull()]

# Remove rows with missing 'default' values from the original data
df = df.dropna(subset=['default'])

predict_data.drop(["worst_status_active_inv","account_worst_status_12_24m","account_worst_status_6_12m",
         "account_incoming_debt_vs_paid_0_24m","account_worst_status_3_6m","account_status",
        "account_worst_status_0_3m","avg_payment_span_0_3m"], axis=1, inplace=True)

# Check the shape of predict_data
print(predict_data.shape)
# Drop the 'uuid' from the actual data. It has no use for prediction part since it is the primary key of the table.
df = df.drop('uuid', axis=1)
# Check the shape of original data
print(df.shape)

# Since the missing values are significant or occur in columns with a high percentage (>=49), mentioned columns are removed from the datasets
df.drop(["worst_status_active_inv", "account_worst_status_12_24m", "account_worst_status_6_12m",
         "account_incoming_debt_vs_paid_0_24m", "account_worst_status_3_6m", "account_status",
         "account_worst_status_0_3m", "avg_payment_span_0_3m"], axis=1)

# Put the remaining attributes into a new object so the analysis part would be easier
# I will use imputation later for the remaining attributes since they are important for prediction and relatively small in number compared to the above columns.
missing_val_col = ["avg_payment_span_0_12m", "num_active_div_by_paid_inv_0_12m", "num_arch_written_off_12_24m",
                   "num_arch_written_off_0_12m",
                   "account_days_in_rem_12_24m", "account_days_in_term_12_24m", "account_days_in_dc_12_24m"]

# Plot histograms for missing_val_col variables
# Identify skewness, outliers, and the general shape of the data
df[missing_val_col].hist(bins=20, figsize=(12, 8))
plt.show()

# When I examined the shape of the distribution, I have seen that it is in lef-skewed (negative skew) form. If it
# were normally distributed, I would prefer to use mean imputation to fill in the missing data but since it is in a
# skewed form, median imputation might be more beneficial.

# See a detailed version of missing_val_col
print(df[missing_val_col].describe())
# Create a pie chart and see the distribution of values (0's and 1's) of default variable
default_counts = df['default'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Default")
plt.axis('equal')
plt.show()


# After I've seen the pie chart, I noticed that the distribution is 98.6% sided to 0. Based on this, I will calculate the medians of 0 and 1 values separately and fill them accordingly.
# CHECK THE FILL MISSING FUNCTION!

def fill_missing():
    for col in missing_val_col:
        median_values = df.groupby('default')[col].median()  # Calculate median values for each default category
        default_0_median = median_values[0]  # Median value for default = 0
        default_1_median = median_values[1]  # Median value for default = 1

        # Fill missing values based on default category
        df.loc[(df['default'] == 0) & (df[col].isnull()), col] = default_0_median   # 0's
        df.loc[(df['default'] == 1) & (df[col].isnull()), col] = default_1_median   # 1's


fill_missing()
print(df[missing_val_col].isnull().sum())
# Missing values in missing_val_col are filled

## HANDLING OUTLIERS ##

# Numerical attributes remaining after deducting values with more than 49% lost value
num_columns_new = ["account_amount_added_12_24m", "account_days_in_dc_12_24m", "account_days_in_rem_12_24m",
                   "account_days_in_term_12_24m", "age", "avg_payment_span_0_12m","max_paid_inv_0_12m",
                   "max_paid_inv_0_24m","num_active_div_by_paid_inv_0_12m", "num_active_inv", "num_arch_dc_0_12m",
                   "num_arch_dc_12_24m","num_arch_ok_0_12m", "num_arch_ok_12_24m",
                   "num_arch_rem_0_12m", "num_arch_written_off_0_12m", "num_arch_written_off_12_24m",
                   "num_unpaid_bills", "recovery_debt", "sum_capital_paid_account_0_12m",
                   "sum_capital_paid_account_12_24m", "sum_paid_inv_0_12m", "time_hours"]

# Categorical attributes remaining after deducting values with more than 49% lost value
cat_columns_new = ["merchant_category","merchant_group","name_in_email","status_last_archived_0_24m","status_2nd_last_archived_0_24m","status_3rd_last_archived_0_24m",
              "status_max_archived_0_6_months","status_max_archived_0_12_months","status_max_archived_0_24_months"]


# Box plot for num_columns_new to see outliers. Interpret it!

"""
for col in num_columns_new:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[col])
    plt.title(f'Boxplot of {col} before IQR and winsorization')
    plt.show()
"""

# Identify outliers using the IQR method: Calculate the IQR (Inter Quartile Range) for each numerical variable and use
# it to identify the potential outliers.

outlier_threshold = 1.5  # Adjust this value to define the threshold for outliers
winsorization_percentile = 0.05  # Adjust this value to define the percentile for winsorization

outliers = []
for col in num_columns_new:
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - outlier_threshold * IQR
    upper_bound = Q3 + outlier_threshold * IQR

    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers.extend(col_outliers.index)

outliers = list(set(outliers))

for col in num_columns_new:
    lower_percentile = winsorization_percentile
    upper_percentile = 1 - winsorization_percentile

    lower_bound = np.percentile(df[col], lower_percentile * 100)
    upper_bound = np.percentile(df[col], upper_percentile * 100)

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

"""
for col in num_columns_new:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[col])
    plt.title(f'Boxplot of {col} after IQR and winsorization')
    plt.show()
"""
## MODEL PART

# Splitting the Dataset
X = df[num_columns_new]  # Features
y = df['default']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression is chosen
model = LogisticRegression(max_iter=1000)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

## XGBoost testing

xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

#XGBoost gave better results
final_model = xgb_classifier

predict_data.drop(["merchant_category","merchant_group","name_in_email","status_last_archived_0_24m","status_2nd_last_archived_0_24m","status_3rd_last_archived_0_24m",
              "status_max_archived_0_6_months","status_max_archived_0_12_months","status_max_archived_0_24_months","has_paid", "default"], axis=1, inplace=True)

print(predict_data.shape)

# Make predictions using the trained XGBoost model
predictions = final_model.predict_proba(predict_data.drop('uuid', axis=1))[:, 1]  # Get the probability of default

# Create a DataFrame with uuid and prob columns
output_df = pd.DataFrame({'uuid': predict_data['uuid'], 'prob': predictions})

# Save the predictions to a CSV file named predictions_final-result
output_df.to_csv("predictions_final-result.csv", index=False)

#############################################################################

## API part

app = Flask("DenizbankProject")



@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the necessary features from the data
    features = data['features']

    # Preprocess the features if required

    # Make predictions using the loaded model
    predictions = model.predict(features)

    # Return the predictions as JSON response
    response = {'predictions': predictions.tolist()}
    return jsonify(response)

if "DenizbankProject" == '__main__':
    app.run()


"""
# Define the API endpoint URL
url = "http://localhost:5000/api/predict"  # Replace with your actual API endpoint URL

# Send the GET request
response = requests.get(url)

# Check the response status code
if response.status_code == 200:
    # Request was successful
    data = response.json()  # Get the response data as JSON
    print(data)
else:
    # Request was unsuccessful
    print("Request failed with status code:", response.status_code)
"""




