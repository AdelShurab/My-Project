import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load data
data = pd.read_csv(r'C:\Users\LENOVO\Saleem_Ai\Saleem_Ai\FeMale & Single (Training).csv')

y = data.iloc[:, 1]  # The second column
X = data.drop(data.columns[1], axis=1)  # All columns except the second column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest classifier
random = RandomForestClassifier()
random.fit(X_train, y_train)

# Save the trained model
dump(random, r'C:\Users\LENOVO\Saleem_Ai\Saleem_Ai\FeMale_Single_model.joblib')

print("Model trained and saved!")
