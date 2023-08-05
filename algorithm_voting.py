# Financial news topic modeling
# Voting Version

# Packages
import numpy as np
#from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Loading data
news = load_files('datasets', encoding='utf-8', decode_error='replace')

# Separating input and output variables
x = news.data
y = news.target

# CONTROL VARIABLES
RANDOM_STATE = 75
TEST_SIZE = .2

# Stop words
#stop_words = set(stopwords.words('english'))

# Split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', norm=None, max_features=1000, decode_error='ignore')

# Applying the Vectorization
x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)

## Building 3 models with 3 different algorithm

model_1 = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs', random_state=30)
model_2 = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=1)
model_3 = MultinomialNB()

results = []

# Initiating the voting model
voting_model = VotingClassifier(estimators=[('LogReg', model_1), ('RandomForest', model_2), ('Multinomial', model_3)], voting='soft')
print('\nVoting Model\n')
print(voting_model)

# Training
voting_model = voting_model.fit(x_train_vectors, y_train)

# Predicts with the test data
predicts = voting_model.predict(x_test_vectors)

# Save the result
results.append(accuracy_score(y_test, predicts))

# Show
print(f'\nAccuracy model: {accuracy_score(y_test, predicts)}\n')
print('\n')