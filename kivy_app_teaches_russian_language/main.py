from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
import pyttsx3
import json
import random
import threading

class RussianLearningApp(App):
    def __init__(self):
        super().__init__()
        self.tts_engine = pyttsx3.init()
        self.current_word = None
        self.score = 0
        self.total_questions = 0
        
    def build(self):
        self.root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(text='Russian Language Learning', size_hint_y=0.1, font_size=24)
        self.root.add_widget(title)
        
        # Score display
        self.score_label = Label(text='Score: 0/0', size_hint_y=0.1)
        self.root.add_widget(self.score_label)
        
        # Word display
        self.word_label = Label(text='Click "New Word" to start', size_hint_y=0.2, font_size=20)
        self.root.add_widget(self.word_label)
        
        # Pronunciation button
        self.pronounce_btn = Button(text='Pronounce', size_hint_y=0.1)
        self.pronounce_btn.bind(on_press=self.pronounce_word)
        self.root.add_widget(self.pronounce_btn)
        
        # Translation input
        self.translation_input = TextInput(hint_text='Enter English translation', size_hint_y=0.1, multiline=False)
        self.root.add_widget(self.translation_input)
        
        # Button layout
        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=10)
        
        new_word_btn = Button(text='New Word')
        new_word_btn.bind(on_press=self.new_word)
        btn_layout.add_widget(new_word_btn)
        
        check_btn = Button(text='Check Answer')
        check_btn.bind(on_press=self.check_answer)
        btn_layout.add_widget(check_btn)
        
        self.root.add_widget(btn_layout)
        
        # Result display
        self.result_label = Label(text='', size_hint_y=0.2)
        self.root.add_widget(self.result_label)
        
        return self.root
    
    def load_words(self):
        try:
            with open('words.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "привет": "hello",
                "спасибо": "thank you",
                "пожалуйста": "please",
                "да": "yes",
                "нет": "no",
                "хорошо": "good",
                "плохо": "bad",
                "вода": "water",
                "еда": "food",
                "дом": "house"
            }
    
    def new_word(self, instance):
        words = self.load_words()
        self.current_word = random.choice(list(words.keys()))
        self.word_label.text = f'Russian: {self.current_word}'
        self.translation_input.text = ''
        self.result_label.text = ''
    
    def pronounce_word(self, instance):
        if self.current_word:
            def speak():
                self.tts_engine.say(self.current_word)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
    
    def check_answer(self, instance):
        if not self.current_word:
            self.result_label.text = 'Please select a new word first!'
            return
        
        words = self.load_words()
        correct_translation = words[self.current_word].lower().strip()
        user_translation = self.translation_input.text.lower().strip()
        
        self.total_questions += 1
        
        if user_translation == correct_translation:
            self.score += 1
            self.result_label.text = f'Correct! ✓'
        else:
            self.result_label.text = f'Incorrect. The answer is: {correct_translation}'
        
        self.score_label.text = f'Score: {self.score}/{self.total_questions}'

if __name__ == '__main__':
    RussianLearningApp().run()