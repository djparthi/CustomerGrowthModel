import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample business data
data = {
    'ad_spend': [1000, 1500, 2000, 2500, 3000],
    'email_signups': [150, 200, 250, 300, 400],
    'new_customers': [30, 45, 50, 60, 75]
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['ad_spend', 'email_signups']]
y = df['new_customers']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict new customer count for new data
new_data = pd.DataFrame({'ad_spend': [3200], 'email_signups': [450]})
prediction = model.predict(new_data)

print(f"Predicted new customers: {int(prediction[0])}")