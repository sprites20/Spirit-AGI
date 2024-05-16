from kivy.config import Config
# Set the window size (resolution)
Config.set('graphics', 'width', '720')
Config.set('graphics', 'height', '1600')

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.graphics import Rectangle, Color
from datetime import datetime, timedelta

import google.generativeai as genai
import re

from openai import OpenAI

import sys
from io import StringIO
from pathlib import Path

# Set up the Cohere API key
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.cohere import Cohere
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

import os
import re


#Retrieve Code Documentation
API_KEY = "BXyTrgsV2PMbRuvDYu9ZwfLHObeTkR4SvPoZAvtf"

# Create the embedding model
embed_model = CohereEmbedding(
    cohere_api_key=API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

# Create the service context with the Cohere model for generation and embedding model
service_context = ServiceContext.from_defaults(
    llm=Cohere(api_key=API_KEY, model="command"),
    embed_model=embed_model
)
user_prompt = "Function to print nth fibonacci number."

TOGETHER_API_KEY = "5391e02b5fbffcdba1e637eada04919a5a1d9c9dfa5795eafe66b6d464d761ce"

client = OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1',
)

genai.configure(api_key='AIzaSyDc0qXo8TxNxv7xAksgwdWtP0Fl1ai6heg')
model = genai.GenerativeModel('gemini-pro')

use_logo = "bot"
KV = '''
<CustomComponent>:
    background_color: (0.5, 0.5, 0.5, 1)
    orientation: 'horizontal'
    size_hint_y: None  # Ensure the height is fixed based on the CustomLabel
    height: custom_label.height # Set the height of CustomComponent based on CustomLabel height
    img_source: ''
    txt: ''
    canvas.before:
        Color:
            rgba: 0.25, 0.25, 0.25, 1  # Background color
        Rectangle:
            pos: self.pos
            size: self.size
    Image:
        source: root.img_source
        size_hint: .1, None  # Disable size_hint so we can set fixed sizes
        size: 50, 50  # Set a fixed size for the image
        pos_hint: {'center_x': 0.5, 'top': 1}  # Position the image at the top of CustomComponent
        canvas.before:
            Color:
                rgba: 0.25, 0.25, 0.25, 1  # Background color
            Rectangle:
                pos: self.x, self.y - (root.height - 50)
                size: 100, root.height
    CustomLabel:
        id: custom_label  # Add an id to the CustomLabel for referencing
        text: root.txt
        height: self.texture_size[1] + 10
        on_release: app.label_clicked(self)
        on_long_press: app.show_popup()
<CustomImageComponent>:
    background_color: (0.5, 0.5, 0.5, 1)
    orientation: 'horizontal'
    size_hint: .1, None
    height: custom_label.height # Set the height of CustomComponent based on CustomLabel height
    img_source: ''
    txt: ''
    canvas.before:
        Color:
            rgba: 0.25, 0.25, 0.25, 1  # Background color
        Rectangle:
            pos: self.pos
            size: self.size
    Image:
        source: "images/empty.png"
        size_hint: 0.1, None  # Disable size_hint so we can set fixed sizes
        size: 50, 50  # Set a fixed size for the image
        pos_hint: {'center_x': 0.5, 'top': 1}  # Position the image at the top of CustomComponent
        canvas.before:
            Color:
                rgba: 0.25, 0.25, 0.25, 1  # Background color
            Rectangle:
                pos: self.pos
                size: self.size
    BoxLayout:
        orientation: 'vertical'
        Image:
            id: custom_label
            source: root.img_source
            size_hint: None, None
            size: max(self.parent.width / 2, 50), max(self.parent.width / 2, 50)
            pos_hint: {'x': 0, 'top': 1}
            keep_ratio: True
            allow_stretch: True
            canvas.before:
                Color:
                    rgba: 0.25, 0.25, 0.25, 1
                Rectangle:
                    pos: self.pos
                    size: self.size
 

<CustomLabel>:
    size_hint_y: None
    height: self.texture_size[1]
    text_size: self.width, None
    #padding: [10,0]
    canvas.before:
        Color:
            rgba: 0.25, .25, .25, 1  # Background color
        Rectangle:
            pos: self.pos
            size: self.size
    markup: True  # Enable markup for custom formatting
    
<TransparentBoxLayout@BoxLayout>:
    canvas.before:
        Color:
            rgba: .25, .25, .25, 1  # Set the color to transparent
        Rectangle:
            pos: self.pos
            size: self.size

MDBoxLayout:
    orientation: 'vertical'
    padding: [0,0,0,100]
    ScrollView:
        canvas.before:
            Color:
                rgba: 0.25, 0.25, 0.25, 1  # Background color
            Rectangle:
                pos: self.pos
                size: self.size

        GridLayout:
            id: grid_layout
            cols: 1
            size_hint_y: None
            height: self.minimum_height
            spacing: '10dp'
            #padding: '30dp'
            
            CustomComponent:
                img_source: "images/bot_logo.png"
                txt: "[b]Bot[/b] [size=12][color=#A9A9A9]{}[/color][/size]\\nHello, how can I help you today?".format(app.current_date)
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: self.minimum_height
        TransparentBoxLayout:
            orientation: 'horizontal'
            
            size_hint_y: None
            height: self.minimum_height
            padding: dp(10)  # Add padding to the TextInput
            TextInput:
                id: text_input
                hint_text: "Type here..."  # Placeholder text
                size_hint_y: None
                size_hint_x: .9
                height: min(self.minimum_height + 25, dp(300))  # Set a maximum height of 100 density-independent pixels
                multiline: True
                pos_hint: {"top": 1}  # Position the TextInput at the top of the TransparentBoxLayout
                background_normal: ''  # Remove the default background
                background_active: ''  # Remove the default background
                background_color: .3, .3, .3, 1  # Set the background color to red with 50% opacity
                foreground_color: 1, 1, 1, 1  # Set the text color to white
        TransparentBoxLayout:
            #orientation: 'vertical'
            size_hint_x: .2
            TransparentBoxLayout:
                MDIconButton:
                    pos_hint: {'center_x': .5, 'center_y': 0.5}  # Center the MDIconButton initially
                    size_hint: None, None
                    size: 50, 50
                    padding_x: 10
                    #pos: self.parent.center_x - self.width / 2, self.parent.center_y - self.height / 2  # Position the MDIconButton at the center of its parent
                    on_release: app.button_pressed()  # Define the action to be taken when the button is released
                    Image:
                        source: "images/gemini_logo.png"
                        size_hint: None, None
                        size: 50, 50  # Set a fixed size for the Image
                        pos_hint: {'center_x': 0.5, 'center_y': 0.5}  # Center the Image initially
                MDIconButton:
                    pos_hint: {'center_x': .5, 'center_y': 0.5}  # Center the MDIconButton initially
                    size_hint: None, None
                    size: 50, 50
                    padding_x: 10
                    #pos: self.parent.center_x - self.width / 2, self.parent.center_y - self.height / 2  # Position the MDIconButton at the center of its parent
                    on_release: app.button_pressed()  # Define the action to be taken when the button is released
                    Image:
                        source: "images/camera_icon.png"
                        size_hint: None, None
                        size: 50, 50  # Set a fixed size for the Image
                        pos_hint: {'center_x': 0.5, 'center_y': 0.5}  # Center the Image initially
            
'''
class CustomComponent(BoxLayout):
    pass

class CustomImageComponent(BoxLayout):
    def __init__(self, img_source="", **kwargs):
        super(CustomImageComponent, self).__init__(**kwargs)
        self.orientation = "horizontal"
        
        # Example usage of img_source and txt as attributes
        self.img_source = img_source
        
        # Example usage of img_source and txt as properties
        # self.add_widget(Image(source=img_source))
        # self.add_widget(Label(text=txt))
    
class CustomLabel(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.long_press_timeout = 0.5  # Set the long press timeout to 1 second
        self.register_event_type('on_long_press')

    def on_long_press(self):
        pass
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._touch = touch
            Clock.schedule_once(self._do_long_press, self.long_press_timeout)
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if hasattr(self, '_touch') and self._touch is touch:
            Clock.unschedule(self._do_long_press)
            del self._touch
        return super().on_touch_up(touch)

    def _do_long_press(self, *args):
        self.dispatch('on_long_press')
    pass

class TransparentBoxLayout(BoxLayout):
    pass
class ChatBoxApp(MDApp):
    # Initialize an empty list to store past messages
    past_messages = []
    def build(self):
        return Builder.load_string(KV)
    
    @property
    def current_date(self):
        # Get the current date and time
        now = datetime.now()
        # Get the current date
        today = now.date()
        # Get yesterday's date
        yesterday = today - timedelta(days=1)
        # Check if the current date is today or yesterday
        if today.day == now.day:
            return now.strftime("Today %I:%M %p")  # %I for 12-hour clock, %p for AM/PM
        elif yesterday.day == now.day:
            return now.strftime("Yesterday %I:%M %p")
        else:
            return now.strftime("%m/%d/%y %I:%M %p")
        

    def gemini_parse_message(self, message):
        formatted_text = message.replace("\\n", "\n")
        formatted_text = formatted_text.replace("\\", "")
        return formatted_text
    
    #Function to initialize conversation
    def start_conversation():
        self.add_message("system", "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them.")
        #add_message("user", "Hello!")
    
    # Function to add a message to the list
    def add_message(self, role, content):
        self.past_messages.append({"role": role, "content": content})
        if role == "user":
            print(f"User: {content}")
    
    def generate_documentation(self, code_path):
        message = []
        code_message = ""
        message.append({"role": "system", "content": "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them."})
        
        file_path = code_path  # Specify the path to your file
        with open(file_path, "r") as file:
            code_message = file.read()
        file_name = Path(file_path).name
        params = f"Code Name: {file_name}\nCode Path: {code_path}"
        file = file_name.replace(".py", "")
        importation = f"To import this module do:\nimport sys\nsys.path.append(\"{file_path}\") #Add the utils' directory to the Python path\n\nThen:\nimport {file}   #Now you can import the module as usual\n\n"
        
        doc_format = """
        Code Location: {code_location} # Full path to the code

        Code Name: {code_name}

        Functions:

        Function Name: {function_name_1}
        Description: {function_description_1}
        Input Arguments:
        {input_args_1}
        Output Arguments:
        {output_args_1}
        Example Usage:
        {function_name_1}({input_args_1})
        Function Name: {function_name_2}
        Description: {function_description_2}
        Input Arguments:
        {input_args_2}
        Output Arguments:
        {output_args_2}
        Example Usage:
        {function_name_2}({input_args_2})
        """
        gen_message = f"Create documentation for this code with given parameters.\n\nParameters:{params}\n\nCode:\n{code_message}\n\nIn this format: \n{doc_format}"
        message.append({"role": "user", "content": gen_message})
        chat_completion = client.chat.completions.create(
          messages=message,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        
        response = chat_completion.choices[0].message.content
        response = response.replace("\\_", "_")

        # Print the assistant's response
        print("Bot: ", response)
        file_path = f"{file_name}_doc.txt"  # Specify the path to your file
        text_to_write = importation + response

        with open(file_path, "w") as file:
            file.write(text_to_write)
    
    # Function to continue the conversation
    def continue_conversation(self):
        #print(past_messages)
        # Create the chat completion request with updated past messages
        chat_completion = client.chat.completions.create(
          messages=self.past_messages,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        
        response = chat_completion.choices[0].message.content
        # Update the past messages list with the new chat completion
        response = response.replace("\\_", "_")
        self.add_message("assistant", response)

        # Print the assistant's response
        print("Bot: ", response)
        return response
    def retrieve_code(self, user_prompt):
        file_path = "somefibo.py_doc.txt"  # Specify the path to your file
        with open(file_path, "r") as file:
            text = file.read()

        # Load the data from the saved chunk files
        data = SimpleDirectoryReader(input_dir="chunk_texts").load_data()

        # Create the index
        index = VectorStoreIndex.from_documents(data, service_context=service_context)
        print(index)

        # Create the Cohere reranker
        cohere_rerank = CohereRerank(api_key=API_KEY)

        # Create the query engine
        query_engine = index.as_query_engine(node_postprocessors=[cohere_rerank])

        # Generate the response
        response = query_engine.query(f"{user_prompt}. Include the location of the program, name, function name, and others if any")

        print(response)
        return response
    def extract_python_code(self, text):
        # Regular expression to match the Python code block
        pattern = r"```python\n(.*?)\n```"

        # Find all matches of the pattern in the input text
        matches = re.findall(pattern, input_text, re.DOTALL)

        # Extract the Python code block from the matches
        python_code = matches[0] if matches else None

        # Print the extracted Python code
        print(python_code)
        return python_code
        
    
    def prepare_code(self, input_text):
        # Input text containing the Python code block
        
        generate_code = f"Retrieved data:\n{retrieved}\nCreate a program based on user prompt. You can import modules based on the retrieved text.\nPrompt:\n{user_prompt}"
        self.add_message("system", generate_code)
        chat_completion = client.chat.completions.create(
          messages=self.past_messages,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        
        response = chat_completion.choices[0].message.content
        response = response.replace("\\_", "_")
        
        code = self.extract_python_code(response)

        # Print the assistant's response
        print("Bot: ", response)
        return code
        
    def get_instruct_type(self, input_text):
        # Input text containing the Python code block
        
        generate_code = f"User Input: {input_text}\nInstruct Types:\n0: Normal, normal conversation\n1: Generate Image, if user wants to generate an image, output only the number of the instruct type, with format: \nFormat: instruct type:<number>"
        message_array = []
        #message_array.append({"role": "system", "content": "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them."})
        message_array.append({"role": "user", "content": generate_code})
        chat_completion = client.chat.completions.create(
          messages=message_array,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        
        response = chat_completion.choices[0].message.content
        print(response)
        # Use regular expression to find the instruct_type number
        pattern = re.compile(r'instruct type\s*:\s*(\d+)')
        # Convert the string to all lowercase
        response = response.lower()
        match = pattern.search(response)
        instruct_type = None
        if match:
            instruct_type = int(match.group(1))
            print(f'instruct_type number: {instruct_type}')
        else:
            print('instruct_type not found in the data')
        #instruct_type = int(response)
        # Print the assistant's response
        #print("Bot: ", response)
        return instruct_type
    
    def generate_image_prompt(self, bot_prompt):
        generate_code = f"User Input: {bot_prompt}\nGenerate an image generation prompt based on user input, with format: \nFormat: image prompt:<prompt>"
        message_array = []
        #message_array.append({"role": "system", "content": "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them."})
        message_array.append({"role": "user", "content": generate_code})
        chat_completion = client.chat.completions.create(
          messages=message_array,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        
        response = chat_completion.choices[0].message.content
        print(response)
        # Regular expression pattern to find the image prompt
        pattern = re.compile(r'image prompt:\s*(.*?)(?:\n|$)', re.IGNORECASE | re.DOTALL)
        #pattern = re.compile(r'instruct type\s*:\s*(\d+)')
        response = response.lower()
        match = pattern.search(response)

        if match:
            image_prompt = match.group(1).strip()
            print(f'Image prompt: {image_prompt}')
        else:
            print('Image prompt not found in the text')
            
        self.generate_image(image_prompt)
        self.add_message("system", f"System: You've successfully generated an image for the user, the generated image prompt: {image_prompt}, you will just say to the user that here's the image.")
    def generate_image(self, prompt):
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "authorization": f"Bearer sk-Y7VUr9D0ikr7fjin6MBb1oAVTHO8eOHjqTGUDpScLbddBih6",
                "accept": "image/*"
            },
            files={"none": ''},
            data={
                "prompt": prompt,
                "output_format": "jpeg",
            },
        )

        if response.status_code == 200:
            with open("images/generated_image.jpeg", 'wb') as file:
                file.write(response.content)
        else:
            raise Exception(str(response.json()))
            
    def run_code(self, code):
        # Capture the standard output
        sys_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        # Execute the Kivy language code
        try:
            exec(code, globals())  # Use globals() to ensure Kivy classes are accessible
        except Exception as e:
            # Display any exception that occurs during execution
            self.root.ids.output_text.text = str(e)
            return
        finally:
            # Restore the original standard output
            sys.stdout = sys_stdout
        print(captured_output.get_value())
        return captured_output.get_value()
            
    def button_pressed(self):
        text_input = self.root.ids.text_input
        
        user_text = text_input.text
        
        #self.generate_documentation("somefibo.py")
        use_model = "together"
        if use_model == "gemini":
            response = model.generate_content(user_text)
            result = str(response._result)
            parsed_result = self.gemini_parse_results(result)
            
            gemini_text = self.gemini_parse_message(parsed_result["text"])
            
            user_header_text = '[b]User[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(self.current_date)
            gemini_header_text = '[b]Gemini[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(self.current_date)
            
            user_message = user_header_text + '\n' + user_text
            gemini_message = gemini_header_text + '\n' + gemini_text
            #print(text_input)
            
            user_custom_component = CustomComponent(img_source="images/user_logo.png", txt=user_message)
            gemini_custom_component = CustomComponent(img_source="images/gemini_logo.png", txt=gemini_message)
            
            grid_layout = self.root.ids.grid_layout
            grid_layout.add_widget(user_custom_component)
            grid_layout.add_widget(gemini_custom_component)
        if use_model == "together":
            # Add a new user message
            self.add_message("user", user_text)
            # Search documents with cohere rerank
            # From web
            # From file
            # In coherererank code
            
            # Decides whether to Run Agent "Code Executor"
            """Decide whether to run the following agents:
            code_executor=<True|False>
            
            """
            # Creates code and runs code
            """
            Generates code with imports and then prepares code for execution
            Executes the code
            """
            """
            #Generate code here
            
            code = 
            output = run_code(code)
            
            """
            
            #self.retrieve_code(user_text)
            #self.prepare_code()
            #self.run_code()
            
            # Code outputs to output.txt
            
            #In json, append the input text, the output.txt into the system txt. And whatever the system does
            #self.add_message("system", system_text)
            
            
            #Instruct Type
            instruct_type = self.get_instruct_type(user_text)
            if instruct_type == 1:
                self.generate_image_prompt(user_text)
            # Continue the conversation            
            response = self.continue_conversation()
            #bot_text = response
            
            user_header_text = '[b]User[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(self.current_date)
            bot_header_text = '[b]Bot[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(self.current_date)
            
            user_message = user_header_text + '\n' + user_text
            bot_message = bot_header_text + '\n' + response
            
            
            #print(text_input)
            
            user_custom_component = CustomComponent(img_source="images/user_logo.png", txt=user_message)
            bot_custom_component = CustomComponent(img_source="images/bot_logo.png", txt=bot_message)
            
            grid_layout = self.root.ids.grid_layout
            grid_layout.add_widget(user_custom_component)
            #grid_layout.add_widget(CustomImageComponent(img_source="images/bug.png"))
            grid_layout.add_widget(bot_custom_component)
            if instruct_type == 1:
                grid_layout.add_widget(CustomImageComponent(img_source="images/generated_image.jpeg"))
        
        
    def label_clicked(self, label):
        print(f"Label '{label.text}' clicked!")
        
    def show_popup(self):
        # Create a popup with buttons
        popup = Popup(title='Actions', size_hint=(None, None), size=(200, 200))
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(Button(text='Action 1'))
        layout.add_widget(Button(text='Action 2'))
        popup.content = layout
        popup.open()
        
    def gemini_parse_results(self, data):
        text = ""
        try:
            text_match = re.search(r'text: "(.*?)"', data, re.DOTALL)
            if text_match:
                text = text_match.group(1)
        except:
            pass
        # Extract the safety ratings
        safety_ratings_match = re.findall(r'category: (.*?)\s+probability: (.*?)\s+}', data, re.DOTALL)
        print(safety_ratings_match)
        
        safety_ratings = [{'category': category.strip(), 'probability': probability.strip()} for category, probability in safety_ratings_match]
        
        # Construct the dictionary
        result = {
            'text': text,
            'safety_ratings': safety_ratings
        }
        return result
if __name__ == '__main__':
    ChatBoxApp().run()