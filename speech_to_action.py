import pyttsx3
import speech_recognition as sr 
import pyaudio
import pygame
import os

class AI():
    __name = ""
    __skill = []

    def __init__(self, name=None):
        self.engine = pyttsx3.init()
        self.r = sr.Recognizer()
        self.m = sr.Microphone()

        if name is not None:
            self.__name = name 

        print("Activating")
        with self.m as source:
            self.r.adjust_for_ambient_noise(source)

    @property 
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        sentence = "Hello, my name is" + self.__name
        self.__name = value
        self.engine.say(sentence)
        self.engine.runAndWait()

    def say(self, sentence):
        self.engine.say(sentence)
        self.engine.runAndWait()

    def listen(self):
        while True:
            print("Say Something")
            with self.m as source:
                audio = self.r.listen(source)
            print("Processing")
            try:
                result = self.r.recognize_google(audio, show_all=True, language="en_US")
                phrase = result["alternative"][0]["transcript"]
                sentence = "Got it, you said" + phrase
                self.engine.say(sentence)
                self.engine.runAndWait()
            except Exception as e:
                print("Sorry, didn't catch that",e)
                self.engine.say("Sorry didn't catch that. Could you please repeat it?")
                self.engine.runAndWait()
            print("Got it, you said '", phrase, "'")
            return phrase
    

face_detected = True
person = "Will"
anomaly = False

if face_detected:
    sentence = " Hello " + person + ", Give me a command."
    instance = AI()
    instance.say(sentence)

    command = instance.listen()

    if command == "play a lullaby":

        instance.say("Playing lullaby")

        lullaby_path = os.path.join(os.getcwd(), "lullaby.mp3")
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load and play the lullaby
        pygame.mixer.music.load(lullaby_path)
        pygame.mixer.music.play()
        
        # Wait until the lullaby finishes playing
        while pygame.mixer.music.get_busy():
            command = instance.listen()
            if command == "stop":
                pygame.mixer.music.stop()
                break

    if command == "provide update":
        if anomaly:
            print("Anomaly detected, check footage")
            instance.say("Anomaly detected, check footage")
        else:
            print("No anomaly detected")
            instance.say("No anomaly detected")