# Import the libraries we need
import pandas as pd                                      # for loading and handling the CSV data
from sklearn.tree import DecisionTreeClassifier          # the decision tree model
from sklearn.model_selection import train_test_split     # splits data into training and test sets
from sklearn.metrics import accuracy_score               # measures how many predictions were correct

# Load the dataset from the CSV file (r'' keeps Windows backslashes literal)
music_data = pd.read_csv(r'c:\Data mining\test_for_ML\music.csv')

# X = input features (every column except 'genre', e.g. age and gender)
X = music_data.drop(columns=['genre'])

# Y = the target we want to predict (the 'genre' column)
Y = music_data['genre']

# Split the data: 80% for training, 20% for testing
# The split is random, so results can change on every run
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Create an empty decision tree model
model = DecisionTreeClassifier()

# Train the model on the training data only
model.fit(X_train, Y_train)

# Predict genres for the test inputs (data the model has never seen)
predictions = model.predict(pd.DataFrame(X_test))

# Compare predictions to the real genres: 1.0 = all correct, 0.0 = none correct
score = accuracy_score(Y_test, predictions)

# Show the accuracy and the predicted genres
print(score)
print(predictions)