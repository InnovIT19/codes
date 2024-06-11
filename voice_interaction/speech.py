import json
import random
import pickle
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
import os
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('voice_data.json') as file:
    intents = json.load(file)['intents']

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def speak(text):
    """Convert text to speech and play it."""
    if not text:
        print("No text to speak")
        return
    tts = gTTS(text=text, lang='en')
    filename = f"{time.time()}.mp3"
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.mixer.music.unload()
    pygame.mixer.quit()
    os.remove(filename)

def recognize_speech():
    """Listen to the user's input and return it as text."""
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            print(f"You said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you repeat?")
            return None
        except sr.RequestError:
            print("Sorry, I'm having trouble understanding you.")
            return None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json, user_input):
    tag = ints[0]['intent']
    list_of_intents = intents_json
    response = "Sorry, I didn't understand that."
    for i in list_of_intents:
        if i['tag'] == tag:
            print(f"Intent found: {tag}")
            if user_input in i['positive_patterns']:
                response = random.choice(i['positive_responses'])
            elif user_input in i['negative_patterns']:
                response = random.choice(i['negative_responses'])
            break
    return response

def main():
    try:
        speak("Hello! How can I assist you today?")
        while True:
            user_input = recognize_speech()
            if user_input:
                if user_input.lower() in ["exit", "quit", "bye"]:
                    speak("You're welcome! It's my pleasure to help you.You're welcome! It's my pleasure to help you.")
                    break

                ints = predict_class(user_input, model)
                if ints:
                    response = get_response(ints, intents, user_input)
                    print(f"Response: {response}")
                    speak(response)
                else:
                    speak("Sorry, I didn't understand that. Could you please rephrase?")
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        pygame.quit()

if __name__ == "__main__":
    main()
