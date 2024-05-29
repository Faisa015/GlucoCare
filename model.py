import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv('diabetes.csv')
df['Outcome'] = df['Outcome'].replace({1:'احتمالية الاصابة',0:'غير مصاب'})
# Select independent and dependent variable
X = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df['Outcome']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = LogisticRegression()

# Fit the model
classifier.fit(X_train, y_train)

# Save the model and the scaler to pickle files
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))