import json
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
intents = json.loads(open('intents.json').read())

# Initialize lists for training data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the pattern and its associated tag to the documents list
        documents.append((word_list, intent['tag']))
    # Add the intent tag to the classes list
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize and remove duplicate words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sort the classes list
classes = sorted(set(classes))

# Save the words and classes lists for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training_sentences = []
training_labels = []

# Create a list of bag-of-words for each pattern and its associated intent
for doc in documents:
    sentence_words = doc[0]  # Tokenized words in the sentence
    training_sentences.append(" ".join(sentence_words))
    
    # Create an output array (one-hot encoded) for the class
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    training_labels.append(output_row)

# Vectorize the training sentences using bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1
    return np.array(bag)

trainX = np.array([bag_of_words(sentence) for sentence in training_sentences])
trainY = np.array(training_labels)

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=300, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')
print('Model training complete and saved!')
print("done")

