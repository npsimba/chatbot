import random
import json
import nltk
import numpy as np
import pickle
from tensorflow.keras.models import load_model  # type: ignore
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import requests
from io import BytesIO

# Load necessary data
lemmatizer = WordNetLemmatizer()
intents_json = json.loads(open('intents.json').read())  # Load intents.json
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Function to clean up user input and create a bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# GUI-related functions
def send_message():
    user_input = user_entry.get()
    if user_input.strip() != "":
        chat_window.config(state=tk.NORMAL)  # Enable the chat window to edit
        chat_window.insert(tk.END, "You: " + user_input + '\n', 'user')  # Add user message
        chat_window.yview(tk.END)  # Scroll to the bottom
        
        # Get chatbot response
        intents = predict_class(user_input)
        response = get_response(intents, intents_json)
        
        chat_window.insert(tk.END, "Bot: " + response + '\n', 'bot')  # Add bot message
        chat_window.config(state=tk.DISABLED)  # Disable chat window to prevent editing

        user_entry.delete(0, tk.END)  # Clear input field



# Create the main window
root = tk.Tk()

# Set the background color (optional)
root.config(bg="#282828")

# Download the image from URL
url = "https://i.pinimg.com/736x/9a/11/33/9a1133d4af3b637e1c6c8ff251785f27.jpg"
response = requests.get(url)
bg_image_data = response.content

# Open the image from the downloaded bytes
bg_image = Image.open(BytesIO(bg_image_data))

# Resize the image to fit the full screen
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()),Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label to display the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Cover the entire window

# Create a chat window (scrollable text area)
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, state=tk.DISABLED, bg="#F2F2F2", fg="black", font=("Arial", 14))
chat_window.grid(row=0, column=0, padx=20, pady=20, columnspan=2)

# Add custom tags for message color styling
chat_window.tag_configure('user', foreground="blue", font=("Arial", 14, 'bold'))
chat_window.tag_configure('bot', foreground="green", font=("Arial", 14, 'bold'))

# Create an entry field for user input
user_entry = tk.Entry(root, width=50, font=("Arial", 14), bg="#FFFFFF", fg="black", bd=5)
user_entry.grid(row=1, column=0, padx=20, pady=20)

# Create a button to send the message
send_button = tk.Button(root, text="Send", width=15, height=2, command=send_message, font=("Arial", 14), bg="#4CAF50", fg="white", bd=5)
send_button.grid(row=1, column=1, padx=10, pady=20)

# Start the Tkinter event loop
root.mainloop()

# Function to toggle full screen on/off
def toggle_full_screen(event=None):
    root.attributes('-fullscreen', not root.attributes('-fullscreen'))
    return "break"

# Function to exit full screen
def quit_full_screen(event=None):
    root.attributes('-fullscreen', False)
    return "break"
