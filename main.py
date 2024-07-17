"""
Done
Fill in the comments
Generate specific nodes
Global storage of node initialization
Global storage of initialized nodes to take positions from
Global storage of connection infos
Global storage of lines
"""

"""
After this generate async function connection string
Else generate node for each async node instantiated
"""

"""
And then web search agent
And local agent
Then the AI thingy, add an OCR agent and map agent
"""


"""
NOW:

Save node inits

Load node inits
Rebind UI components with their names from tree view on_run_press_wrapper
By iterating through button nodes and binding with their

"""

"""
NOW:

Add more nodes
Facebook search
Google Search
Image Search
Vectara RAG
STT
LLava

Map Processor
Reverse Geocoding
Geocoding
Place Search

Map Integration

And UI for multiagent
Screenshot Node

Transpiler

Node Optimizer
"""

from kivy.config import Config

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.behaviors import DragBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivymd.uix.button import MDIconButton
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.treeview import TreeView, TreeViewLabel
from kivy.uix.popup import Popup
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST, glCullFace, GL_BACK
from kivy.graphics import RenderContext, Callback, PushMatrix, PopMatrix, \
    Color, Translate, Rotate, Mesh, UpdateNormalMatrix, BindTexture
from kivy.uix.codeinput import CodeInput
from kivy.properties import Property
from objloader import ObjFile
from tkinter import Tk, filedialog
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.properties import NumericProperty
from kivy.core.audio import SoundLoader
import pyttsx3

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

from pyproj import Proj, transform
from pygments.lexers import PythonLexer

from textblob import TextBlob

from openai import OpenAI

from io import StringIO
from pathlib import Path

from datetime import datetime, timedelta

#import google.generativeai as genai
from io import StringIO

"""
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
"""
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

import sys
import time
import asyncio
import json
import numpy as np
import os
import re
import requests
import threading
import copy

import pytesseract
from PIL import Image
import cv2

import ast

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import cohere

#Retrieve Code Documentation
API_KEY = "BXyTrgsV2PMbRuvDYu9ZwfLHObeTkR4SvPoZAvtf"

cohere_api_key = "5VkxjOzk1Jo4nkonX5XPM7ks9rHIm0h9lPacWQMx"
"""
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
"""
user_prompt = "Function to print nth fibonacci number."

TOGETHER_API_KEY = "5391e02b5fbffcdba1e637eada04919a5a1d9c9dfa5795eafe66b6d464d761ce"

client = OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1',
)

MISTRAL_API_KEY = "VuNE7EzbFp5QA0zoYl0LokvrTitF7yrg"
client_mistral = MistralClient(api_key=MISTRAL_API_KEY)

"""
genai.configure(api_key='AIzaSyDc0qXo8TxNxv7xAksgwdWtP0Fl1ai6heg')
model = genai.GenerativeModel('gemini-pro')
"""
use_logo = "bot"


lines = {}
lines_canvas = {}
connections = {}
nodes = {}
node_info = {}

global_touch = None
global_drag = False
added_node = False

language_codes = None
with open('language_codes.json') as json_file:
    language_codes = json.load(json_file)
        
nodes_regenerated = 0

def is_point_in_ellipse(point, center, size):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    a = size[0]
    b = size[1]
    return (dx*dx) / (a*a) + (dy*dy) / (b*b) <= 1


#Note
"""
We gotta exec the funciton string and import string
"""

def is_module_imported(module_name):
    return module_name in sys.modules

def generate_documentation(code_path):
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

class NewNodeScreen(Screen):
    def __init__(self, **kwargs):
        super(NewNodeScreen, self).__init__(**kwargs)
        self.current_documentation = None
        self.current_description = None
        self.current_code = None
        self.current_inputs = None
        self.current_outputs = None
        self.current_name = None
        
        self.selected = None
        
        main_scroll = ScrollView()
        main_layout = BoxLayout(orientation='vertical')
        # Create the back button layout
        back_box = BoxLayout(size_hint=(1, None), height=40)
        back_button = Button(text="Back", on_press=self.switch_to_screen)
        switch_button = Button(text="Code Interpreter")
        save_button = Button(text="Save Node", on_press=self.save_the_node)
        #back_button.bind(on_press=self.back_button_on_press)
        back_box.add_widget(back_button)
        back_box.add_widget(switch_button)
        back_box.add_widget(save_button)
        main_layout.add_widget(back_box)
        
        name_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        name_label = Label(text="Name", size_hint_x = .25)
        self.name_input = TextInput(size_hint_x = .75)
        #back_button.bind(on_press=self.back_button_on_press)
        name_box.add_widget(name_label)
        name_box.add_widget(self.name_input)
        main_layout.add_widget(name_box)
        # Parameters label
        params_label_box = BoxLayout(size_hint=(1, None), height=40)
        main_layout.add_widget(params_label_box)
        
        
        # Bbox layout
        bbox_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=150)
        
        # Scrollable input
        input_box = BoxLayout(orientation='vertical', size_hint=(0.5, 1))
        input_label_text = Label(text="Inputs")
        input_label_box = BoxLayout(size_hint=(1, None), height=50)
        input_label_box.add_widget(input_label_text)
        input_box.add_widget(input_label_box)
        
        self.input_textinput = TextInput(size_hint=(1, None), height=150, hint_text='Example:\n"addend_1" : "num",\n"addend_2" : "num"\n')
        input_box.add_widget(self.input_textinput)
        
        # Scrollable outputs
        output_box = BoxLayout(orientation='vertical', size_hint=(0.5, 1))
        output_label_text = Label(text="Outputs")
        output_label_box = BoxLayout(size_hint=(1, None), height=50)
        output_label_box.add_widget(output_label_text)
        output_box.add_widget(output_label_box)
        
        self.output_textinput = TextInput(size_hint=(1, None), height=150, hint_text='Example:\n"sum" : "num"\n')
        output_box.add_widget(self.output_textinput)
        
        bbox_layout.add_widget(input_box)
        bbox_layout.add_widget(output_box)
        
        main_layout.add_widget(bbox_layout)
        
        button_select_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        main_layout.add_widget(button_select_box)
        
        self.description_button = Button(text="Description", on_release=self.switch_to_description)
        button_select_box.add_widget(self.description_button)
        
        self.documentation_button = Button(text="Documentation", on_release=self.switch_to_documentation)
        button_select_box.add_widget(self.documentation_button)
        
        self.code_button = Button(text="Code", on_release=self.switch_to_code)
        button_select_box.add_widget(self.code_button)
        

        # Description Text Box
        self.description_scroll = ScrollView(size_hint=(1, 0.5))
        self.description_textinput = TextInput(size_hint=(1, None), multiline=True)
        
        # Define a function to calculate height
        def calculate_height(instance, value):
            return instance.minimum_height + 1000
        
        # Bind the height of the TextInput to its minimum_height + additional_height
        self.description_textinput.bind(minimum_height=lambda instance, value: setattr(self.description_textinput, 'height', calculate_height(instance, value)))
        
        # Add the TextInput to the ScrollView
        self.description_scroll.add_widget(self.description_textinput)
        main_layout.add_widget(self.description_scroll)
        
        button_generate_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        main_layout.add_widget(button_generate_box)
        
        self.description_gen_button = Button(text="Generate from Description", on_release=self.generate_from_description)
        button_generate_box.add_widget(self.description_gen_button)
        
        self.documentation_gen_button = Button(text="Generate from Documentation", on_release=self.generate_from_documentation)
        button_generate_box.add_widget(self.documentation_gen_button)
        
        self.code_gen_button = Button(text="Generate from Code", on_release=self.generate_from_code)
        button_generate_box.add_widget(self.code_gen_button)
        
        # Scrollable input
        user_feedback_box = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50)
        input_label_text = Label(text="User Feedback", size_hint_x=0.75)
        send_feedback_button = Button(text="Send Feedback", size_hint_x=0.25)

        user_feedback_box.add_widget(input_label_text)
        user_feedback_box.add_widget(send_feedback_button)
        main_layout.add_widget(user_feedback_box)
        
        self.feedback_textinput = TextInput(size_hint=(1, None), multiline=True, hint_text="Enter your feedback here. Can be a bug, an error, a modification, a fix, etc.")
        
        main_layout.add_widget(self.feedback_textinput)
        #main_scroll.add_widget(main_layout)
        self.add_widget(main_layout)
        
        self.switch_to_description(instance = None)
    
    def save_the_node(self, instance):
        #Update node_init
        print("Saving...")
        data = {
            "function_name": self.current_name,
            "import_string" : None,
            "function_string" : self.current_code or self.description_textinput.text,
            "description" : self.current_description or self.description_textinput.text,
            "documentation" : self.current_documentation or self.description_textinput.text,
            "inputs" : ast.literal_eval(f"{{{self.input_textinput.text}}}"),
            "outputs": ast.literal_eval(f"{{{self.output_textinput.text}}}")
        }
        
        global node_init
        node_init[self.current_name] = data
        print(node_init)
        #Save node_init
        print("Node Init: ")
        f = open("node_init.json", "w")
        f.write(json.dumps(node_init))
        f.close()
    
    def generate_from_description(self, instance):
        if self.selected == "documentation":
            self.generate_documentation(instance = None, context=self.current_description, gen_from="description")
        elif self.selected == "code":
            self.generate_code(instance = None, context=self.current_description, gen_from="description")
        else:
            self.current_description = self.description_textinput.text
            
    def generate_from_documentation(self, instance):
        if self.selected == "description":
            self.generate_description(instance = None, context=self.current_documentation, gen_from="documentation")
        elif self.selected == "code":
            self.generate_code(instance = None, context=self.current_documentation, gen_from="documentation")
        else:
            self.current_documentation = self.description_textinput.text
        
    def generate_from_code(self, instance):
        if self.selected == "description":
            self.generate_description(instance = None, context=self.current_code, gen_from="code")
        elif self.selected == "documentation":
            self.generate_documentation(instance = None, context=self.current_code, gen_from="code")
        else:
            self.current_code = self.description_textinput.text
            
    def generate_code(self, instance, context, gen_from):
        message = []
        new_message = ChatMessage(role = "system", content = "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them.")
        message.append(new_message)
        
        inputs = self.input_textinput.text or ""
        outputs = self.output_textinput.text or ""
        self.current_name = self.name_input.text or ""
        
        doc_format = f"""
async def {self.current_name}(node, {{input1}}, {{input2}}): #Remember to change based on Inputs.
        #Insert Function
        return {{{{\"output1\"}} : "", {{\"output2\"}} : []}} #Remember to change based on Outputs.
        """
        
        gen_message = f"Generate python code given parameters.\n\nInputs:\n{inputs}\n\nOutputs:\n{outputs}\n\n{gen_from}:\n{context}\n\nIn this format: \n{doc_format}\n\nMake sure to replace inputs and outputs with the given names. Output nothing else but the code with indentations. Enclose with ```python"
        new_message = ChatMessage(role = "user", content = gen_message)
        message.append(new_message)
        """
        chat_completion = client.chat.completions.create(
          messages=message,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        """

        model = "codestral-latest"
        chat_completion = client_mistral.chat(
            model=model,
            messages=message
        )
        
        response = chat_completion.choices[0].message.content
        response = response.replace("\\_", "_")
        response = response.replace(" ```python", "")
        response = response.replace("```python", "")
        response = response.replace("```", "")
        # Print the assistant's response
        print("Bot: ", response)
        
        self.current_code = response
        self.description_textinput.text = response
            
    def generate_description(self, instance, context, gen_from):
        message = []
        new_message = ChatMessage(role = "system", content = "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them.")
        message.append(new_message)
        inputs = self.input_textinput.text or ""
        outputs = self.output_textinput.text or ""
        self.current_name = self.name_input.text or ""
        
        doc_format = f"""
Example:

Function Name: add
This function adds 2 numbers and returns a sum.
It is used when asked:
- What is the sum of numbers 1 and 2?
- The sum of 1 and 2 is?
        """
        
        gen_message = f"Generate short description code given parameters.\n\nInputs:\n{inputs}\n\nOutputs:\n{outputs}\n\n{gen_from}:\n{context}\n\nIn this format: \n{doc_format}\n\n. Output nothing else but the description."
        new_message = ChatMessage(role = "user", content = gen_message)
        message.append(new_message)
        """
        chat_completion = client.chat.completions.create(
          messages=message,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        """

        model = "codestral-latest"
        chat_completion = client_mistral.chat(
            model=model,
            messages=message
        )
        
        response = chat_completion.choices[0].message.content
        response = response.replace("\\_", "_")

        # Print the assistant's response
        print("Bot: ", response)
        
        self.current_description = response
        self.description_textinput.text = response
        
    def generate_documentation(self, instance, context, gen_from):
        message = []
        code_message = ""
        new_message = ChatMessage(role = "system", content = "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them.")
        message.append(new_message)
        
        inputs = self.input_textinput.text or ""
        outputs = self.output_textinput.text or ""
        
        doc_format = """
        Node Name: {code_name}
        
        Functions:

        Function Name: {function_name_1}
        Description: {function_description_1}
        Input Arguments:
        {input_args_1}
        Output Arguments:
        {output_args_1}
        Example Usage:
        {function_name_1}({input_args_1})
        """
        
        
        gen_message = f"Generate documentation of code given parameters. Base on what the code does\n\nInputs:{inputs}\n\nOutputs:{outputs}\n\n{gen_from}:\n{context}\n\nIn this format: \n{doc_format}"
        new_message = ChatMessage(role = "user", content = gen_message)
        message.append(new_message)
        """
        chat_completion = client.chat.completions.create(
          messages=message,
          model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        """

        model = "codestral-latest"
        chat_completion = client_mistral.chat(
            model=model,
            messages=message
        )
        
        response = chat_completion.choices[0].message.content
        response = response.replace("\\_", "_")

        # Print the assistant's response
        print("Bot: ", response)
        
        self.current_documentation = response
        self.description_textinput.text = response
    
    def switch_to_screen(self, instance):
        # Switch to 'chatbox'
        self.manager.transition = NoTransition()
        self.manager.current = 'draggable_label_screen'
        
    def switch_to_description(self, instance):
        if self.selected == "documentation":
            self.current_documentation = self.description_textinput.text or ""
        elif self.selected == "code":
            self.current_code = self.description_textinput.text or ""
        self.selected = "description"
        self._switch_input(TextInput, self.current_description or "", hint_text= "This is the description.")
        self.update_button_colors(self.description_button)

    def switch_to_documentation(self, instance):
        if self.selected == "description":
            self.current_description = self.description_textinput.text or ""
        elif self.selected == "code":
            self.current_code = self.description_textinput.text or ""
        self.selected = "documentation"
        self._switch_input(TextInput, self.current_documentation or "", hint_text= "This is the documentation.")
        self.update_button_colors(self.documentation_button)

    def switch_to_code(self, instance):
        if self.selected == "description":
            self.current_description = self.description_textinput.text or ""
        elif self.selected == "documentation":
            self.current_documentation = self.description_textinput.text or ""
        self.selected = "code"
        self._switch_input(CodeInput, self.current_code or "", hint_text= "print('This is the code')", lexer=PythonLexer())
        self.update_button_colors(self.code_button)
    
    def _switch_input(self, input_type, text, **kwargs):
        
        # Remove the current text input
        self.description_scroll.remove_widget(self.description_textinput)
        # Create a new text input of the given type
        self.description_textinput = input_type(size_hint=(1, None), multiline=True, **kwargs)
        self.description_textinput.text = text
        # Bind the height to adjust based on content
        self.description_textinput.bind(minimum_height=lambda instance, value: setattr(self.description_textinput, 'height', instance.minimum_height + 1000))
        # Add the new text input to the scroll view
        self.description_scroll.add_widget(self.description_textinput)

    def update_button_colors(self, active_button):
        buttons = [self.description_button, self.documentation_button, self.code_button]
        for button in buttons:
            if button == active_button:
                button.background_color = [1, 0, 0, 1]  # Red
            else:
                button.background_color = [1, 1, 1, 1]  # White
            
class Renderer(Widget):
    def __init__(self, **kwargs):
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.source = resource_find('simple.glsl')
        self.scene = ObjFile(resource_find("output.obj"))
        
        super(Renderer, self).__init__(**kwargs)

        with self.canvas:
            Color(1, 1, 1, 1)  # Set background color to white
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
            print("Canvas Drawn")
            
        Clock.schedule_interval(self.update_glsl, 1 / 60.)
        
        # Create a layout to hold the button
        layout = FloatLayout(size=self.size)
        self.add_widget(layout)

        # Create a button and add it to the layout
        self.button = Button(text='Click me!', size_hint=(None, None), size=(100, 50), pos=(10, self.height - 60))
        self.button.bind(on_release=self.button_callback)
        layout.add_widget(self.button)

    def button_callback(self, instance):
        # Callback function for the button
        print("Button clicked!")

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)
        
    def update_glsl(self, delta):
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['texture0'] = 1
        self.canvas['projection_mat'] = proj
        self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.rot.angle += delta * 30

    def setup_scene(self):
        meshes = list(self.scene.objects.values())  # Get all meshes from the scene
        print(meshes)
        for mesh in meshes:
            BindTexture(source='Earth_TEXTURE_CM.tga', index=1)
            Color(1, 1, 1, 1)
            PushMatrix()
            Translate(0, 0, -15)
            self.rot = Rotate(1, 0, 1, 0)
            UpdateNormalMatrix()
            self.mesh = Mesh(
                vertices=mesh.vertices,
                indices=mesh.indices,
                fmt=mesh.vertex_format,
                mode='triangles',
            )
            PopMatrix()

class RenderScreen(Screen):
    def __init__(self, **kwargs):
        super(RenderScreen, self).__init__(**kwargs)
        layout = FloatLayout()
        top_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, pos_hint={'top': 1})
        back_button = Button(text='Back', on_press=self.switch_to_screen)
        top_layout.add_widget(back_button)
        # Add the 3D renderer to the layout
        renderer = Renderer(size_hint=(1, 0.8))
        layout.add_widget(renderer)

        # Add a box on top of the renderer
        box = Button(text='Box', size_hint=(0.2, 0.1), pos_hint={'x': 0.4, 'y': 0.1})
        layout.add_widget(top_layout)
        layout.add_widget(box)
        

        self.add_widget(layout)
    def switch_to_screen(self, instance):
        # Switch to 'chatbox'
        self.manager.transition = NoTransition()
        self.manager.current = 'draggable_label_screen'
        
class AsyncNode:
    def __init__(self, function_name=None, node_id=None, input_addresses=[], output_args={}, trigger_out=[]):
        self.trigger_in = None
        self.trigger_out = []
        self.function_name = function_name
        self.input_addresses = input_addresses
        self.output_args = output_args
        self.node_id = node_id
        self.stop = False
        self.args = {}
        
    def change_to_red(self):
        nodes[self.node_id].label_color.rgba = (1,0,0,1)
    def change_to_gray(self):
        nodes[self.node_id].label_color.rgba = (0.5, 0.5, 0.5, 1)
    async def trigger_node(self, node):
        node.trigger_in = self.node_id
        print(node, node.trigger_in)
        await node.trigger()
    async def trigger(self):
        if self.trigger_in:
            if self.trigger_in.startswith("stop_after"):
                self.stop = True
            elif self.trigger_in.startswith("reset_outputs"):
                print(f"Resetting output {self.node_id}")
                try:
                    for arg_name in self.output_args:
                        self.output_args[arg_name] = None
                except:
                    pass
                return None
        # Get the function from the dictionary based on the function_name
        function_to_call = functions.get(self.function_name)
        print(self, self.node_id)
        if function_to_call:
            #print(f"Calling function {self.function_name}")
            # Fetch input_args from input_addresses
            input_args = {}
            print(self.input_addresses)
            for address in self.input_addresses:
                
                node = address.get("node")
                arg_name = address.get("arg_name")
                target = address.get("target")
                try:
                    input_args[target] = node.output_args.get(arg_name)
                except Exception as e:
                    print(node, arg_name, target, e)
                print(input_args[target])
                #Here replace thing in output args with whatever queued. If none use same thing
            #print("Input Addresses: ", self.input_addresses)
            #print("Input Args", input_args)
            # Pass input_args and self to the function
            # Schedule UI update in the main Kivy thread
            Clock.schedule_once(lambda dt: self.change_to_red(), 0)
            output_args = await function_to_call(self, **input_args)
            Clock.schedule_once(lambda dt: self.change_to_gray(), 0)
            print("Output args: ", output_args)

            # Update output_args with the function's output, appending new args and replacing existing ones
            try:
                for arg_name, value in output_args.items():
                    if arg_name not in self.output_args:
                        self.output_args[arg_name] = value
                    else:
                        self.output_args[arg_name] = value
            except:
                pass
        #print(node)
        print("Output args: ", self.output_args)
        #print(self.output_args)
        """
        for node in self.trigger_out:
            #print(f"Triggering output node {node.function_name}")
            await node.trigger()
        """
        if not self.stop:
            tasks = [asyncio.create_task(self.trigger_node(node)) for node in self.trigger_out]
            await asyncio.gather(*tasks)
        else:
            self.stop = False

class MousePositionWidget(Widget):
    def __init__(self, **kwargs):
        super(MousePositionWidget, self).__init__(**kwargs)
        self.prev_pos = None
        #self.total_dx = 0  # Total delta x
        #self.total_dy = 0  # Total delta y

    def on_touch_move(self, touch):
        if self.prev_pos:
            dx = touch.pos[0] - self.prev_pos[0]
            dy = touch.pos[1] - self.prev_pos[1]
            #print("Mouse delta:", (dx, dy))
            
            if not any(isinstance(child, DraggableLabel) and child.dragging for child in self.parent.children):
                # If no box is being dragged, update total delta
                #self.total_dx += dx
                #self.total_dy += dy
                # Move all boxes by the total delta
                for child in self.parent.children:
                    if isinstance(child, DraggableLabel):
                        child.pos = (child.pos[0] + dx, child.pos[1] + dy)
        
        self.prev_pos = touch.pos
        return super(MousePositionWidget, self).on_touch_move(touch)
    def on_touch_up(self, touch):
        self.prev_pos = None

class TouchableRectangle(Widget):
    def __init__(self, **kwargs):
        super(TouchableRectangle, self).__init__(**kwargs)
        self.bind(pos=self.update_rect, size=self.update_rect)
        with self.canvas.before:
            Color(0.3, 0.3, 0.3, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.line2 = None
        self.line = None
        self.curr_i = None
        self.collide_mode = None
    def update_rect(self, *args):
        # Update the position of the existing Rectangle
        self.rect.pos = self.pos

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            parent = self.parent
            parent.dragging = True  # Set dragging to True when touch is on the label
            #print("Touched TouchableRectangle")
            #print("Touched DraggableLabel:", self.name)

            if not hasattr(touch, 'touchdown'):
                touch.touchdown = True
                #print(parent.name)
                for i in parent.output_label_circles:
                    #print(i, parent.output_label_circles[i].pos, touch.pos)
                    if (parent.output_label_circles[i].pos[0] <= touch.pos[0] <= parent.output_label_circles[i].pos[0] + 10 and
                        parent.output_label_circles[i].pos[1] <= touch.pos[1] <= parent.output_label_circles[i].pos[1] + 10):
                        # Change the circle color when held
                        #print(True)
                        self.collide_mode = 0
                        self.curr_i = i
                        #print(self.curr_i)
                        with self.canvas:
                            Color(1, 0, 0)
                            self.line = Line(points=[parent.output_label_circles[self.curr_i].pos[0] + 5, parent.output_label_circles[self.curr_i].pos[1] + 5, *touch.pos])
                        return super(TouchableRectangle, self).on_touch_down(touch)
                for i in parent.input_label_circles:
                    #print(i, parent.output_label_circles[i].pos, touch.pos)
                    if (parent.input_label_circles[i].pos[0] <= touch.pos[0] <= parent.input_label_circles[i].pos[0] + 20 and
                        parent.input_label_circles[i].pos[1] <= touch.pos[1] <= parent.input_label_circles[i].pos[1] + 20):
                        # Change the circle color when held
                        #print(True)
                        #self.curr_i = i
                        self.collide_mode = 1
                        print(parent.node_id, i)
                        print("Connections: ", connections[parent.node_id])
                        print("Collision: ", i)
                        if connections[parent.node_id]["inputs"]:
                            curr_in = connections[parent.node_id]["inputs"]
                            for j in curr_in:
                                curr_j = curr_in[j]
                                print(j, curr_j)
                                for k in curr_j:
                                    curr_k = curr_j[k]
                                    if k == i:
                                        for l in connections[parent.node_id]["inputs"][j][k]:
                                            
                                            #Remove Line
                                            print(lines[connections[parent.node_id]["inputs"][j][k][l]])
                                            print(connections[parent.node_id]["inputs"][j][k][l])
                                            #print(connections[parent.node_id]["inputs"][j][k])
                                            #Remove that connection bidirectionally
                                            print("Node info ", parent.node_id, node_info[parent.node_id]["input_addresses"])
                                            #filtered_data = [item for item in node_info[parent.node_id]["input_addresses"] if item['target'] != i]
                                            node_info[parent.node_id]["input_addresses"] = [item for item in node_info[parent.node_id]["input_addresses"] if item['target'] != i]
                                            #print("Filtered ", node_info[parent.node_id]["input_addresses"])
                                            #print("Async ", parent.node_id, async_nodes[parent.node_id].input_addresses)
                                            async_nodes[parent.node_id].input_addresses = [item for item in async_nodes[parent.node_id].input_addresses if item['target'] != i]
                                            #print("Filtered ", async_nodes[parent.node_id].input_addresses )
                                            # Remove the dictionary where target == 'user_prompt'
                                            
                                            #Remove in async where target == i
                                            #print("Node info ", j, node_info[j])
                                            #print("Async ", j, async_nodes[j])
                                            #Remove in the input_addresses
                                            #async_nodes[curr_child].input_addresses.append({"node": async_nodes[curr_parent], "arg_name": self.curr_i, "target": curr_j})
                                            #node_info[curr_child]["input_addresses"].append({"node": curr_parent, "arg_name": self.curr_i, "target": curr_j})
                                            
                                            #Remove the Line
                                            #print("Lines: ")
                                            lines_canvas[connections[parent.node_id]["inputs"][j][k][l]].remove(lines[connections[parent.node_id]["inputs"][j][k][l]])
                                            del lines[connections[parent.node_id]["inputs"][j][k][l]]
                                            del connections[parent.node_id]["inputs"][j][k]
                                            
                                            
                                            """
                                            print(parent.canvas)
                                            for i in self.canvas.children:
                                                print(i)
                                            """
                                            #print(parent, self, parent.parent)
                                            #lines2 = [item for item in parent.parent.canvas.children if isinstance(item, Line)]
                                            #print(lines2)
                                            #self.canvas.remove(lines[connections[parent.node_id]["inputs"][j][k][l]])
                                            #self.canvas.remove(lines[connections[parent.node_id]["inputs"][j][k][l]])
                                        break
                        #print(self.curr_i)
                        #print(self.curr_i)
                        with self.canvas:
                            Color(1, 0, 0)
                        return super(TouchableRectangle, self).on_touch_down(touch)
        return super(TouchableRectangle, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.line:
            circle_center_x = self.parent.output_label_circles[self.curr_i].pos[0] + 5
            circle_center_y = self.parent.output_label_circles[self.curr_i].pos[1] + 5
            self.line.points = [circle_center_x, circle_center_y, *touch.pos]
        return super(TouchableRectangle, self).on_touch_move(touch)
        
    def on_touch_up(self, touch):
        if self.line:
            parent = self.parent
            parent.dragging = False  # Set dragging to True when touch is on the label
            
            curr_child = None
            curr_parent = None
            curr_j = None
            found_circle = False
            for child in self.parent.parent.children:
                print(child)
                if isinstance(child, DraggableLabel) and child != self:
                    print(child.box_rect.pos, *touch.pos, child.node_id)
                    if child.box_rect.collide_point(*touch.pos):
                        print("Touch collides with", child.node_id)
                        #Check for collision in the outputs of that box only
                        if self.collide_mode == 0:
                            for j in child.input_label_circles:
                                print(j, child.input_label_circles[j].pos)
                                if is_point_in_ellipse(touch.pos, child.input_label_circles[j].pos, (10, 10)):
                                    found_circle = True
                                    print("Found point")
                                    with self.canvas:
                                        Color(1, 0, 0)
                                        #print(self.curr_i)
                                        if(self.curr_i):
                                            #Instantiate line on lines by appending
                                            seconds = time.time()
                                            lines[str(seconds)] = (Line(points=[parent.output_label_circles[self.curr_i].pos[0] + 5, parent.output_label_circles[self.curr_i].pos[1] + 5,
                                                                      child.input_label_circles[j].pos[0] + 5, child.input_label_circles[j].pos[1] + 5]))
                                            #self.connection = (child.input_circle_pos[0] + 5, child.input_circle_pos[1] + 5)
                                            #If collided save the id of the connection bidirectionally, the id of this and the other.
                                            # Initialize connections if it is not already initialized
                                            if parent.node_id not in connections:
                                                connections[parent.node_id] = {
                                                    "inputs" : {},
                                                    "outputs" : {}
                                                }
                                            if child.node_id not in connections:
                                                connections[child.node_id] = {
                                                    "inputs" : {},
                                                    "outputs" : {}
                                                }
                                            
                                            # First get the id of node, store in connections, with value of the line
                                            if child.node_id not in connections[parent.node_id]["outputs"]:
                                                connections[parent.node_id]["outputs"][child.node_id] = {}
                                            if self.curr_i not in connections[parent.node_id]["outputs"][child.node_id]:
                                                connections[parent.node_id]["outputs"][child.node_id][self.curr_i] = {}
                                            connections[parent.node_id]["outputs"][child.node_id][self.curr_i][j] = str(seconds)
                                            # Then get the id of the child, store in connections
                                            
                                            # First get the id of node, store in connections, with value of the line
                                            if parent.node_id not in connections[child.node_id]["inputs"]:
                                                connections[child.node_id]["inputs"][parent.node_id] = {}
                                            if j not in connections[child.node_id]["inputs"][parent.node_id]:
                                                connections[child.node_id]["inputs"][parent.node_id][j] = {}
                                            connections[child.node_id]["inputs"][parent.node_id][j][self.curr_i] = str(seconds)
                                            
                                            #Add connection from node_id to the child_node_id
                                            print(parent.node_id)
                                            curr_parent = str(parent.node_id)
                                            curr_child = str(child.node_id)
                                            curr_j = str(j)
                                            
                                            
                                            #print(async_nodes[parent.node_id].input_addresses)
                                            #print(lines)
                                            #print(parent.node_id, connections[parent.node_id])
                                            #print(child.node_id, connections[child.node_id])
                                            #print(connections)
                                        break
                                    
                                    
            if found_circle:
                async_nodes[curr_child].input_addresses.append({"node": async_nodes[curr_parent], "arg_name": self.curr_i, "target": curr_j})
                node_info[curr_child]["input_addresses"].append({"node": curr_parent, "arg_name": self.curr_i, "target": curr_j})
                
                print(curr_child)
                
                #print("node_info: ", node_info)
                #print("async_nodes: ", async_nodes)
                found_circle = False
            self.canvas.remove(self.line)
            self.line = None
        return super(TouchableRectangle, self).on_touch_up(touch)
        
class TruncatedLabel(Label):
    def on_texture_size(self, instance, size):
        if size[0] > self.width:
            # Calculate the approximate width of "..." based on font size
            ellipsis_width = len("...") * self.font_size / 3
            # Calculate the maximum text width based on Label width and ellipsis width
            max_width = self.width - ellipsis_width
            text = self.text
            self.max_length = 7  # Maximum length of the text
            # Truncate the text to fit within the Label width
            if len(self.text) > self.max_length:
                self.text = self.text[:self.max_length] + "..."
            else:
                self.text = self.text
            """  
            while self.texture_size[0] > max_width and len(text) > 0:
                text = text[:-1]
                self.text = text + "..."
            """
        else:
            # Text fits within the Label width
            self.text = self.text
def display_output_2(user_text):
    app = MDApp.get_running_app()
    print("App: ", app)
    instruct_type = app.get_instruct_type(user_text)
    if instruct_type == 1:
        app.generate_image_prompt(user_text)
    # Continue the conversation            
    response = app.continue_conversation()
    #bot_text = response
    
    user_header_text = '[b]User[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(app.current_date)
    bot_header_text = '[b]Bot[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(app.current_date)
    
    user_message = user_header_text + '\n' + user_text
    bot_message = bot_header_text + '\n' + response
    
    
    #print(text_input)
    
    user_custom_component = CustomComponent(img_source="images/user_logo.png", txt=user_message)
    bot_custom_component = CustomComponent(img_source="images/bot_logo.png", txt=bot_message)
    
    grid_layout = app.root.get_screen("chatbox").ids.grid_layout
    grid_layout.add_widget(user_custom_component)
    #grid_layout.add_widget(CustomImageComponent(img_source="images/bug.png"))
    grid_layout.add_widget(bot_custom_component)
    if instruct_type == 1:
        grid_layout.add_widget(CustomImageComponent(img_source="images/generated_image.jpeg"))

class DraggableLabel(DragBehavior, Label):
    def __init__(self, inputs, name, node_id, outputs, pos, regenerated=False, **kwargs):
        super(DraggableLabel, self).__init__(**kwargs)
        #print("Adding DraggableLabel: ", node_id, "at pos: ", pos)
        self.name = name
        self.node_id = node_id
        self.inputs = inputs
        #print()
        self.outputs = outputs
        self.regenerated = regenerated
        
        self.color = (1, 1, 1, 0)
        
        self.text = self.node_id
        self.size_hint = (None, None)
        self.size_x = 200
        self.size = (self.size_x, 50)
        self.bottom_rect_size = (self.size_x, 20 + 16 * max(len(self.outputs), len(self.inputs)))
        
        self.initialized = False

        self.prev_pos = None  # Previous position of the widget
        self.dragging = False  # Flag to track whether the label is being dragged

        self.offsetted_pos = pos
        self.pos = (pos[0], pos[1])
        self.mouse_offset = [0,0]
        
        self.to_be_deleted = False

        
        with self.canvas.before:
            self.input_labels = {}
            self.input_label_circles = {}
            
            self.output_labels = {}
            self.output_label_circles = {}
            
            self.line = None  # Initialize the line object
            self.line2 = None
            
            self.trigger_lines = {}
            
            self.label = Label(pos=self.pos, size=self.size, text=self.name)
            self.add_widget(self.label)
            
            self.label_color = Color(0.5, 0.5, 0.5, 1)  # Set the label background color (gray in this case)
            self.label_rect = Rectangle(pos=self.pos, size=self.size)
            self.box_color = (0.3, 0.3, 0.3, 1)  # Set the box background color
            # Create a TouchableRectangle as the box_rect
            self.box_rect = TouchableRectangle(pos=self.bottom_rect_size, size=self.bottom_rect_size)
            self.add_widget(self.box_rect)

            # Define the positions of the input and output circles
            self.input_circle_pos = (self.x - 3, self.y + self.height / 2 - 5)
            self.output_circle_pos = (self.right - 7, self.y + self.height / 2 - 5)

            # Draw the input and output circles
            self.input_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.output_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.input_circle = Ellipse(pos=self.input_circle_pos, size=(10, 10))
            self.output_circle = Ellipse(pos=self.output_circle_pos, size=(10, 10))
            
            
        
        with self.canvas.after:
            count = 1
            for i in self.inputs:
                #print(i)
                # Add labels to the bottom box
                max_length = 10  # Maximum length of the text
                text = i
                # Truncate the text to fit within the Label width
                if len(i) > max_length:
                    text = i[:max_length] + "..."
                    
                label = TruncatedLabel(text=f'{text}', size=(dp(len(f'{text}')*5), dp(10)))
                self.add_widget(label)
                self.input_labels[i] = label
                #print(len(f'{i}')*10)
                #print(self.input_labels[i])
                self.input_labels[i].pos = (self.x-10, self.y - self.height - (20 * count))
                
                ellipse_pos = (self.x-22, self.y - self.height - (20 * count))
                ellipse = Ellipse(pos=ellipse_pos, size=(10, 10))
                self.input_label_circles[i] = ellipse
                count += 1
                
            count = 1
            for i in self.outputs:
                # Add labels to the bottom box
                label = TruncatedLabel(text=f'{i}', size=(dp(len(f'{i}')*5), dp(10)))
                self.add_widget(label)
                self.output_labels[i] = label
                #print(self.output_labels[i])
                self.output_labels[i].pos = (self.x-10, self.y - self.height - (20 * count))
                
                ellipse_pos = (self.x+2, self.y - self.height - (20 * count))
                ellipse = Ellipse(pos=ellipse_pos, size=(10, 10))
                self.output_label_circles[i] = ellipse
                count += 1
                
        self.update_rect()

    def update_rect(self, *args):
        self.pos = (self.offsetted_pos[0], self.offsetted_pos[1])
        self.label.pos = (self.offsetted_pos[0], self.offsetted_pos[1])
        self.label_rect.pos = (self.offsetted_pos[0], self.offsetted_pos[1])
        #self.label_rect.pos = 
        self.label_rect.size = self.size
        
        self.box_rect.pos = (self.offsetted_pos[0], self.offsetted_pos[1] - (20 + 16 * max(len(self.outputs), len(self.inputs))))
        self.box_rect.size = (self.width, 20 + 16 * max(len(self.outputs), len(self.inputs)))
        # Update the positions of the input and output circles
        self.input_circle_pos = (self.offsetted_pos[0] - 3, self.offsetted_pos[1] + self.height / 2 - 5)
        self.output_circle_pos = (self.right - 7, self.offsetted_pos[1] + self.height / 2 - 5)
        self.input_circle.pos = self.input_circle_pos
        self.output_circle.pos = self.output_circle_pos
        
        count = 1
        for i in self.inputs:
            self.input_labels[i].pos = (self.offsetted_pos[0]+20, self.offsetted_pos[1] - (20 * count))
            self.input_label_circles[i].pos = (self.offsetted_pos[0]-3, self.offsetted_pos[1] - (20 * count))
            count += 1
        count = 1
        for i in self.outputs:
            self.output_labels[i].pos = (self.offsetted_pos[0] + self.width - self.output_labels[i].width - 20, self.offsetted_pos[1] - (20 * count))
            self.output_label_circles[i].pos = (self.offsetted_pos[0] + self.width-7, self.offsetted_pos[1] - (20 * count))
            count += 1
        
        if self.line2:
            self.line2.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                self.connection[0], self.connection[1]]

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.to_be_deleted:
                #Delete all lines from detected connections from connections
                #Delete thing in canvas
                #Delete trigger_outs by finding in node_info that has that trigger_out
                #Delete node_info
                #Delete connections
                #Delete asyncnode
                pass
            self.dragging = True  # Set dragging to True when touch is on the label
            global_drag = True
            # Check if the touch is within the bounds of the circles
            if (self.output_circle_pos[0] <= touch.pos[0] <= self.output_circle_pos[0] + 20 and
                    self.output_circle_pos[1] <= touch.pos[1] <= self.output_circle_pos[1] + 20):
                # Change the circle color when held
                self.input_circle_color.rgba = (1, 0, 0, 1)  # Red color
                self.output_circle_color.rgba = (1, 0, 0, 1)  # Red color
                # Create a line from circle to touch position
                with self.canvas:
                    Color(1, 0, 0)
                    self.line = Line(points=[self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, *touch.pos])
                return super(DraggableLabel, self).on_touch_down(touch)
                
            if (self.input_circle_pos[0] <= touch.pos[0] <= self.input_circle_pos[0] + 20 and
                    self.input_circle_pos[1] <= touch.pos[1] <= self.input_circle_pos[1] + 20):
                # Change the circle color when held
                #print(True)
                #self.curr_i = i
                self.collide_mode = 1
                i = "input_circle"
                print(self.node_id, i)
                print("Connections: ", connections[self.node_id])
                print("Collision: ", i)
                if connections[self.node_id]["inputs"]:
                    curr_in = connections[self.node_id]["inputs"]
                    for j in curr_in:
                        curr_j = curr_in[j]
                        print(j, curr_j)
                        for k in curr_j:
                            curr_k = curr_j[k]
                            if k == i:
                                for l in connections[self.node_id]["inputs"][j][k]:
                                    
                                    #Remove Line
                                    #print(lines[connections[self.node_id]["inputs"][j][k][l]])
                                    #print(connections[self.node_id]["inputs"][j][k][l])
                                    #print(connections[self.node_id]["inputs"][j][k])
                                    #Remove that connection bidirectionally
                                    #print("Node info ", self.node_id, node_info[self.node_id]["input_addresses"])
                                    #filtered_data = [item for item in node_info[self.node_id]["input_addresses"] if item['target'] != i]
                                    #node_info[self.node_id]["trigger_out"] = [item for item in node_info[self.node_id]["input_addresses"] if item['target'] != i]
                                    #print("Filtered ", node_info[self.node_id]["input_addresses"])
                                    #print("Async ", self.node_id, async_nodes[self.node_id].input_addresses)
                                    #Iterate through every node and find which node that has trigger_out as self.node_id
                                    for q in async_nodes:
                                        #print(q, async_nodes[q].trigger_out)
                                        if async_nodes[self.node_id] in async_nodes[q].trigger_out:
                                            print("Found in: ", q)
                                            async_nodes[q].trigger_out.remove(async_nodes[self.node_id])
                                            node_info[q]["trigger_out"].remove(self.node_id)
                                            print(async_nodes[q].trigger_out)
                                    #print("Async ", async_nodes[self.node_id].trigger_out)
                                    #async_nodes[self.node_id].trigger_out = [item for item in async_nodes[self.node_id].trigger_out if item != i]
                                    #print("Filtered ", async_nodes[self.node_id].input_addresses )
                                    # Remove the dictionary where target == 'user_prompt'
                                    
                                    #Remove in async where target == i
                                    #print("Node info ", j, node_info[j])
                                    #print("Async ", j, async_nodes[j])
                                    #Remove in the input_addresses
                                    #async_nodes[curr_child].input_addresses.append({"node": async_nodes[curr_parent], "arg_name": self.curr_i, "target": curr_j})
                                    #node_info[curr_child]["input_addresses"].append({"node": curr_parent, "arg_name": self.curr_i, "target": curr_j})
                                    
                                    #Remove the Line
                                    #print("Lines: ")
                                    lines_canvas[connections[self.node_id]["inputs"][j][k][l]].remove(lines[connections[self.node_id]["inputs"][j][k][l]])
                                    del lines[connections[self.node_id]["inputs"][j][k][l]]
                                    del connections[self.node_id]["inputs"][j][k]
                                    
                                    
                                    """
                                    print(parent.canvas)
                                    for i in self.canvas.children:
                                        print(i)
                                    """
                                    #print(parent, self, parent.parent)
                                    #lines2 = [item for item in parent.parent.canvas.children if isinstance(item, Line)]
                                    #print(lines2)
                                    #self.canvas.remove(lines[connections[self.node_id]["inputs"][j][k][l]])
                                    #self.canvas.remove(lines[connections[self.node_id]["inputs"][j][k][l]])
                                break
        # Allow dragging of the box
        self.drag_rectangle = (self.x, self.y, self.width, self.height)
        return super(DraggableLabel, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        #print(touch)
        if self.prev_pos:
            # Calculate the delta between the current and previous positions
            dx = touch.pos[0] - self.prev_pos[0]
            dy = touch.pos[1] - self.prev_pos[1]
            
            self.mouse_offset[0] += dx
            self.mouse_offset[1] += dy
            #print("Delta from MousePositionWidget:", self.mouse_offset)
        self.prev_pos = touch.pos
        if self.regenerated:
            temp_pos = (node_info[self.node_id]["pos"][0], node_info[self.node_id]["pos"][1])
            #print("Added Node Move", temp_pos)
            self.pos = (temp_pos[0], temp_pos[1])
            self.label.pos = (temp_pos[0], temp_pos[1])
            self.label_rect.pos = (temp_pos[0], temp_pos[1])
            #self.label_rect.pos = 
            self.label_rect.size = self.size
            
            self.box_rect.pos = (temp_pos[0], temp_pos[1] - (20 + 16 * max(len(self.outputs), len(self.inputs))))
            self.box_rect.size = (self.width, 20 + 16 * max(len(self.outputs), len(self.inputs)))
            # Update the positions of the input and output circles
            self.input_circle_pos = (temp_pos[0] - 3, temp_pos[1] + self.height / 2 - 5)
            self.output_circle_pos = (self.right - 7, temp_pos[1] + self.height / 2 - 5)
            self.input_circle.pos = self.input_circle_pos
            self.output_circle.pos = self.output_circle_pos
            
            count = 1
            for i in self.inputs:
                self.input_labels[i].pos = (temp_pos[0]+20, temp_pos[1] - (20 * count))
                self.input_label_circles[i].pos = (temp_pos[0]-3, temp_pos[1] - (20 * count))
                count += 1
            count = 1
            for i in self.outputs:
                self.output_labels[i].pos = (temp_pos[0] + self.width - self.output_labels[i].width - 10, temp_pos[1] - (20 * count))
                self.output_label_circles[i].pos = (temp_pos[0] + self.width-20, temp_pos[1] - (20 * count))
                count += 1
            
            if self.line2:
                self.line2.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                    self.connection[0], self.connection[1]]
            
            self.regenerated = False
            global nodes_regenerated
            nodes_regenerated += 1
            if nodes_regenerated == len(nodes):
                added_node = False
            #print(self.pos)
        elif self.dragging:
            #print(self.pos)
            offsetted = [self.x, self.y]
            self.pos = (offsetted[0], offsetted[1])
            node_info[self.node_id]["pos"] = offsetted
            self.label.pos = self.pos
            self.label_rect.pos = self.pos
            self.label_rect.size = self.size
            
            self.box_rect.pos = (self.x, self.y - (20 + 16 * max(len(self.outputs), len(self.inputs))))
            self.box_rect.size = (self.width, (20 + 16 * max(len(self.outputs), len(self.inputs))))
            # Update the positions of the input and output circles
            self.input_circle_pos = (self.x - 3, self.y + self.height / 2 - 5)
            self.output_circle_pos = (self.right - 7, self.y + self.height / 2 - 5)
            self.input_circle.pos = self.input_circle_pos
            self.output_circle.pos = self.output_circle_pos
            
            
            count = 1
            for i in self.inputs:
                self.input_labels[i].pos = (self.x+20, self.y - (20 * count))
                self.input_label_circles[i].pos = (self.x-3, self.y - (20 * count))
                count += 1
            count = 1
            for i in self.outputs:
                self.output_labels[i].pos = (self.x + self.width - self.output_labels[i].width - 20, self.y - (20 * count))
                self.output_label_circles[i].pos = (self.x + self.width-7, self.y - (20 * count))
                count += 1
            
            if self.line2:
                self.line2.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                    self.connection[0], self.connection[1]]
        elif not global_drag:
            if not self.initialized:
                offsetted = [self.offsetted_pos[0], self.offsetted_pos[1]]
                self.initialized = True
            #print(self.pos)
            else:
                offsetted = [self.x, self.y]
            self.pos = (offsetted[0], offsetted[1])
            node_info[self.node_id]["pos"] = offsetted
            self.label.pos = self.pos
            self.label_rect.pos = self.pos
            self.label_rect.size = self.size
            
            self.box_rect.pos = (self.x, self.y - (20 + 16 * max(len(self.outputs), len(self.inputs))))
            self.box_rect.size = (self.width, (20 + 16 * max(len(self.outputs), len(self.inputs))))
            
            # Update the positions of the input and output circles
            self.input_circle_pos = (self.x - 3, self.y + self.height / 2 - 5)
            self.output_circle_pos = (self.right - 7, self.y + self.height / 2 - 5)
            self.input_circle.pos = self.input_circle_pos
            self.output_circle.pos = self.output_circle_pos

            
            count = 1
            for i in self.inputs:
                self.input_labels[i].pos = (self.x+20, self.y - (20 * count))
                self.input_label_circles[i].pos = (self.x-3, self.y - (20 * count))
                count += 1
            count = 1
            for i in self.outputs:
                self.output_labels[i].pos = (self.x + self.width - self.output_labels[i].width - 20, self.y - (20 * count))
                self.output_label_circles[i].pos = (self.x + self.width-7, self.y - (20 * count))
                count += 1
            
            if self.line2:
                self.line2.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                    self.connection[0], self.connection[1]]
        
        
        
        if self.line:
            self.line.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, *touch.pos]
        #print(self.node_id, self.node_id in connections)
        #print(connections)
        if self.node_id in connections:
            #print(connections)
            #print(f"Connections of {self.node_id}: ", sum(len(value) for value in connections[self.node_id].values()))
            for i in connections[self.node_id]["outputs"]:
                #print("i: ", i)
                for j in connections[self.node_id]["outputs"][i]:
                    #print("j: ", j)
                    for k in connections[self.node_id]["outputs"][i][j]:
                        curr_line = connections[self.node_id]["outputs"][i][j][k]
                        if j != "output_circle":
                            if curr_line in lines:
                                lines[curr_line].points = [self.output_label_circles[j].pos[0] + 5, self.output_label_circles[j].pos[1] + 5, lines[curr_line].points[2], lines[curr_line].points[3]]
                        else:
                            #pass
                            if curr_line in lines:
                                lines[curr_line].points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, lines[curr_line].points[2], lines[curr_line].points[3]]
            for i in connections[self.node_id]["inputs"]:
                #print("i: ", i)
                for j in connections[self.node_id]["inputs"][i]:
                    #print("j: ", j)
                    for k in connections[self.node_id]["inputs"][i][j]:
                        curr_line = connections[self.node_id]["inputs"][i][j][k]
                        #print(j)
                        if j != "input_circle":
                            if curr_line in lines:
                                lines[curr_line].points = [lines[curr_line].points[0], lines[curr_line].points[1], self.input_label_circles[j].pos[0] + 5, self.input_label_circles[j].pos[1] + 5]
                        else:
                            #pass
                            if curr_line in lines:
                                lines[curr_line].points = [lines[curr_line].points[0], lines[curr_line].points[1], self.input_circle_pos[0] + 5, self.input_circle_pos[1] + 5]
        """
        if self.prev_pos:
            # Calculate the delta between the current and previous positions
            dx = touch.pos[0] - self.prev_pos[0]
            dy = touch.pos[1] - self.prev_pos[1]
            print("Delta from MousePositionWidget:", (dx, dy))
        self.prev_pos = touch.pos
        """
        
        return super(DraggableLabel, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        self.prev_pos = None
        self.dragging = False  # Set dragging to True when touch is on the label
        global_drag = False
        curr_child = None
        curr_parent = None
        found_circle = False
        #print("Children: ", self.parent.children)
        if self.line:
            # Check if any other DraggableLabel instances are colliding with the output circle
            for child in self.parent.children:
                if isinstance(child, DraggableLabel) and child != self:
                    #Make for loop to loop through all other nodes
                    #Check if collides with their box (Optional)
                    #Loop through every input in that node
                    """
                    for input_circle in child.input_label_circles:
                        print(input_circle.pos)
                        #Print key and value
                    """
                    
                    if is_point_in_ellipse(touch.pos, child.input_circle_pos, (10, 10)):
                        found_circle = True
                        print("Found point")
                        # Create a line connecting the output circle of the current instance to the input circle of the other instance
                        with self.canvas:
                            Color(1, 0, 0)
                            #If collided save the id of the connection bidirectionally, the id of this and the other.
                            #First get the id of the child, store in connections
                            #Instantiate line on lines by appending
                            seconds = time.time()
                            lines[str(seconds)] = (Line(points=[self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                                      child.input_circle_pos[0] + 5, child.input_circle_pos[1] + 5]))
                            #self.connection = (child.input_circle_pos[0] + 5, child.input_circle_pos[1] + 5)
                            #If collided save the id of the connection bidirectionally, the id of this and the other.
                            # Initialize connections if it is not already initialized
                            if self.node_id not in connections:
                                connections[self.node_id] = {
                                    "inputs" : {},
                                    "outputs" : {}
                                }
                            if child.node_id not in connections:
                                connections[child.node_id] = {
                                    "inputs" : {},
                                    "outputs" : {}
                                }
                            
                            # First get the id of node, store in connections, with value of the line
                            if child.node_id not in connections[self.node_id]["outputs"]:
                                connections[self.node_id]["outputs"][child.node_id] = {}
                            if "output_circle" not in connections[self.node_id]["outputs"][child.node_id]:
                                connections[self.node_id]["outputs"][child.node_id]["output_circle"] = {}
                            connections[self.node_id]["outputs"][child.node_id]["output_circle"]["input_circle"] = str(seconds)
                            # Then get the id of the child, store in connections
                            
                            # First get the id of node, store in connections, with value of the line
                            if self.node_id not in connections[child.node_id]["inputs"]:
                                connections[child.node_id]["inputs"][self.node_id] = {}
                            if "input_circle" not in connections[child.node_id]["inputs"][self.node_id]:
                                connections[child.node_id]["inputs"][self.node_id]["input_circle"] = {}
                            connections[child.node_id]["inputs"][self.node_id]["input_circle"]["output_circle"] = str(seconds)
                            
                            
                            
                            curr_child = str(child.node_id)
                            curr_parent = str(self.node_id)
                            #async_nodes[str(self.node_id)].trigger_out.append(async_nodes[str(child.node_id)])
                            #print(child.node_id, async_nodes[self.node_id].trigger_out)
                            #Add connection and name
                            #print("Connections: ", connections)
                            
                            
                            #print(self.connection)
                            #print(child.text)
                        
                        #print(child.node_id)
                        #The circle collided in that id, store in connections[id]
                        #print("output_circle")
                        #The id of this node, store in connections
                        #print(self.node_id)
                        #The id of circle of this node, from which is detected from touch_down store in self
                        #print("input_circle")
                        #Create a line globally with id number, use the points of that point and this point
                        #lines[]
                        #Link the line id to the connections
                        break
            if found_circle:
                print(curr_child)
                async_nodes[curr_parent].trigger_out.append(async_nodes[curr_child])
                node_info[curr_parent]["trigger_out"].append(curr_child)
                print("async_nodes: ", async_nodes)
                print("node_info", node_info)
                found_circle = False
            self.canvas.remove(self.line)
            self.line = None
            self.input_circle_color.rgba = (1, 1, 1, 1)  # Gray color
            self.output_circle_color.rgba = (1, 1, 1, 1)  # Gray color
            print("Connections: ", connections)
        return super(DraggableLabel, self).on_touch_up(touch)
image_components = []
node_init = {
    "text_to_wav_instance" : {
        "function_name": "text_to_wav_instance",
        "import_string" : None,
        "function_string" : """
'''
async def text_to_wav_instance(node, text):
    return None
'''

import time
import wave

from TTS.api import TTS
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
except:
    pass
def split_long_sentence(sentence, max_length=250):
    if len(sentence) <= max_length:
        return [sentence]
    
    sentences = []
    while len(sentence) > max_length:
        # Find the last space before max_length
        last_space_idx = sentence.rfind(' ', 0, max_length)
        # If no space is found, split at max_length
        if last_space_idx == -1:
            last_space_idx = max_length
        sentences.append(sentence[:last_space_idx])
        sentence = sentence[last_space_idx+1:]
    
    if sentence:
        sentences.append(sentence)
    
    return sentences
def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the number of frames in the file
        frames = wav_file.getnframes()
        # Get the frame rate (number of frames per second)
        frame_rate = wav_file.getframerate()
        # Calculate the duration of the file in seconds
        duration = frames / float(frame_rate)
        return duration
async def text_to_wav_instance(node, text):
    filename = "output.wav"
    if tts:
        if node.trigger_in.startswith("prompt"):
            #Split audio first
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)
            split_sentences = []
            for sentence in sentences:
                if len(sentence) > 250:
                    split_sentences.extend(split_long_sentence(sentence))
                else:
                    split_sentences.append(sentence)
            node.args["sentences"] = split_sentences
            
            
        print(node.args["sentences"])
        
        while not node.args["sentences"][0]:
            await asyncio.sleep(0.1)
        

        
        if node.args["sentences"][0]:
            # generate speech by cloning a voice using default settings
            tts.tts_to_file(node.args["sentences"][0],
            file_path="output.wav",
            speaker_wav="audio.wav",
            speed=1,
            language="en")
        else:
            pass
        
        try:
            node.args["sentences"].pop(0)
        except:
            pass
        
        try:
            while node.args["sound"].is_playing():
                await asyncio.sleep(0.1)
                print("Playing")
        except Exception as e:
            print("Error, no sound found", e)
            node.args["sound"] = None
        
        sound = SoundLoader.load(filename)
        duration = get_wav_duration("output.wav")
        
        
        return {"speech_wav" : sound, "duration" : duration}
    else:
        engine = pyttsx3.init()
        engine.save_to_file(text, filename)
        engine.runAndWait()
        
        sound = SoundLoader.load(filename)
        node.args["sound"] = sound
        return {"speech_wav" : sound}

        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "text" : "string"
        },
        "outputs": {
            "speech_wav" : "sound",
            "duration" : "num"
        }
    },
    "play_audio_tts" : {
        "function_name": "play_audio_tts",
        "import_string" : None,
        "function_string" : """
async def play_audio_tts(node, sound, duration):
    if node.trigger_in.startswith("text_to_wav_instance"):
        if not "sounds" in node.args:
            print("Sounds Created")
            node.args["sounds"] = []
            sound.play()
        if not "durations" in node.args:
            print("Durations Created")
            node.args["durations"] = []
        node.args["sounds"].append(sound)
        node.args["durations"].append(duration)
        
    if node.trigger_in.startswith("pass_node"):
        if node.args["sounds"]:
            node.args["sounds"][0].play()
            await asyncio.sleep(node.args["durations"][0])
            
            node.args["sounds"].pop(0)
            node.args["durations"].pop(0)
            
            await asyncio.sleep(2)
            #Delay by audio duration
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "sound" : "sound",
            "duration" : "num"
        },
        "outputs": {
        }
    },
    "stop_audio_tts" : {
        "function_name": "stop_audio_tts",
        "import_string" : None,
        "function_string" : """
async def stop_audio_tts(node, sound):
    pass
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "sound" : "sound"
        },
        "outputs": {
        }
    },
    "file_chooser" : {
        "function_name": "file_chooser",
        "import_string" : None,
        "function_string" : """
async def file_chooser(node):
    print(node, node.node_id, node.output_args)
    if node.trigger_in.startswith("display_output"):
        node.output_args = {"user_image" : None}
        return {"user_image" : None}
    else:
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        def pop(dt):
            popup = Popup(title='No file selected',
                          content=Label(text='No file selected.'),
                          size_hint=(None, None), size=(400, 200))
            popup.open()
        if file_path:
            #self.image.source = file_path
            return {"dir" : file_path}
        else:
            Clock.schedule_once(pop)
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
        },
        "outputs": {
            "dir" : "string"
        }
    },
    "image_chooser" : {
        "function_name": "image_chooser",
        "import_string" : None,
        "function_string" : """
async def image_chooser(node):
    print(node, node.node_id, node.output_args)
    if node.trigger_in.startswith("display_output"):
        node.output_args = {"user_image" : None}
        return {"user_image" : None}
    else:
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        root.destroy()
        def pop(dt):
            popup = Popup(title='No file selected',
                          content=Label(text='No file selected.'),
                          size_hint=(None, None), size=(400, 200))
            popup.open()
        if file_path:
            #self.image.source = file_path
            return {"user_image" : file_path}
        else:
            Clock.schedule_once(pop)
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
        },
        "outputs": {
            "user_image" : "string"
        }
    },
    "ignition" : {
            "function_name": "ignition",
            "import_string" : None,
            "function_string" : """
async def ignition(node):
    print("Ignition")
    await asyncio.sleep(.25)
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
            },
            "outputs": {
            }
        },
    "display_output" : {
        "function_name": "display_output",
        "import_string" : None,
        "function_string" : """
async def display_output(node, user_input, output, instruct_type, generated_image_path, user_image):
    app = MDApp.get_running_app()
    print("Display Output: ", user_input, output)
    user_text = user_input or "test"
    response = output or "test"
    await asyncio.sleep(.25)
    def update_ui(dt):
        user_header_text = '[b]User[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(app.current_date)
        bot_header_text = '[b]Bot[/b] [size=12][color=#A9A9A9]{}[/color][/size]'.format(app.current_date)
        
        user_message = user_header_text + '\\n' + user_text
        bot_message = bot_header_text + '\\n' + response
        
        user_custom_component = CustomComponent(img_source="images/user_logo.png", txt=user_message)
        bot_custom_component = CustomComponent(img_source="images/bot_logo.png", txt=bot_message)
        
        grid_layout = app.root.get_screen("chatbox").ids.grid_layout
        
        grid_layout.add_widget(user_custom_component)
        print(user_image)
        if user_image != None:
            print(user_image)
            grid_layout.add_widget(CustomImageComponent(img_source=user_image))
        grid_layout.add_widget(bot_custom_component)
        
        if instruct_type == 1:
            #image_components.append(CustomImageComponent(img_source=generated_image_path))
            grid_layout.add_widget(CustomImageComponent(img_source=generated_image_path))

    # Schedule the update_ui function to run on the main thread
    Clock.schedule_once(update_ui)
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "user_input" : "string",
            "output" : "string",
            "instruct_type" : "num",
            "generated_image_path" : "string",
            "user_image" : "string",
        },
        "outputs": {
        }
    },

    "select_model" : {
            "function_name": "select_model",
            "import_string" : None,
            "function_string" : """
async def select_model(node):
    print("select_model")
    await asyncio.sleep(.25)
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
            },
            "outputs": {
                "model" : "string",
            }
        },
    "user_input" : {
            "function_name": "user_input",
            "import_string" : None,
            "function_string" : """
async def user_input(node):
    print("user_input")
    await asyncio.sleep(.25)
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
            },
            "outputs": {
                "user_input" : "string",
            }
        },
    "pass_node" : {
            "function_name": "pass_node",
            "import_string" : None,
            "function_string" : """
async def pass_node(node):
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
            },
            "outputs": {
            }
        },
    "context" : {
            "function_name": "context",
            "import_string" : None,
            "function_string" : """
async def context(node):
    print("context")
    await asyncio.sleep(.25)
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
            },
            "outputs": {
                "context" : "string",
            }
        },
    "reset_outputs" : {
            "function_name": "reset_outputs",
            "import_string" : None,
            "function_string" : """
async def reset_outputs(node):
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
            },
            "outputs": {
            }
        },
    "is_greater_than" : {
        "function_name": "is_greater_than",
        "import_string" : None,
        "function_string" : """
async def is_greater_than(node, A=None, B=None):
    return {"is_greater_than" : A > B}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "A" : "bool",
            "B" : "bool"
        },
        "outputs": {
            "is_greater_than" : "bool"
        }
    },
    "is_less_than" : {
        "function_name": "is_less_than",
        "import_string" : None,
        "function_string" : """
async def is_less_than(node, A=None, B=None):
    return {"is_less_than" : A < B}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "A" : "bool",
            "B" : "bool"
        },
        "outputs": {
            "is_less_than" : "bool"
        }
    },
    "is_equal" : {
        "function_name": "is_equal",
        "import_string" : None,
        "function_string" : """
async def is_equal(node, A=None, B=None):
    return {"is_equal" : A == B}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "A" : "bool",
            "B" : "bool"
        },
        "outputs": {
            "is_equal" : "bool"
        }
    },
    "prompt" : {
        "function_name": "prompt",
        "import_string" : None,
        "function_string" : """
async def prompt(node, model=None, user_prompt=None, context=None):
    app = MDApp.get_running_app()
    print("Prompt")
    print(model, user_prompt, context)
    await asyncio.sleep(.25)
    user_text = user_prompt
    instruct_type = app.get_instruct_type(user_text)
    if context:
        context = "OCR output:\\n" + context
        print("context: ", context)
    generated_image_path = ""
    if instruct_type == 1:
        generated_image_path = app.generate_image_prompt(user_text)
    if instruct_type == 2:
        pass
    # Continue the conversation            
    response = app.continue_conversation(context=context)
    print("output: ", response)
    return {"output" : response, "instruct_type" : instruct_type, "generated_image_path" : generated_image_path}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "model" : "string",
            "user_prompt" : "string", 
            "context" : "string",
        },
        "outputs": {
            "output" : "string",
            "instruct_type" : "num",
            "generated_image_path" : "string",
        }
    },
    "image_to_text" : {
        "function_name": "image_to_text",
        "import_string" : None,
        "function_string" : """
async def image_to_text(node, image_path=None):
    if image_path:
        # Load the image using PIL
        
        image = Image.open(image_path)

        # Convert the image to a format OpenCV can work with
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Use pytesseract to get detailed OCR results
        detailed_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Initialize variables to store sentence/paragraph bounding boxes and text
        boxes = []
        current_box = None
        current_text = ""

        # Loop over each of the text elements found in the image
        for i in range(len(detailed_data['level'])):
            (x, y, w, h) = (detailed_data['left'][i], detailed_data['top'][i], detailed_data['width'][i], detailed_data['height'][i])
            text = detailed_data['text'][i]
            conf = int(detailed_data['conf'][i])
            
            # Only consider text elements with a confidence above a certain threshold
            if conf > 40:
                if current_box is None:
                    # Start a new bounding box and text group
                    current_box = (x, y, x + w, y + h)
                    current_text = text
                else:
                    # Check if the text element is on a new line
                    if y > current_box[3]:
                        # Add a newline character
                        current_text += "\\n"
                    # Expand the current bounding box to include the new text element
                    current_box = (
                        min(current_box[0], x),
                        min(current_box[1], y),
                        max(current_box[2], x + w),
                        max(current_box[3], y + h)
                    )
                    # Append text to the current group
                    current_text += " " + text
                
                # Check if the next element is a new paragraph or sentence (using heuristic)
                if i == len(detailed_data['level']) - 1 or detailed_data['block_num'][i] != detailed_data['block_num'][i + 1]:
                    boxes.append((current_box, current_text))
                    current_box = None
                    current_text = ""
        output_text = ""
        # Draw bounding boxes around sentences/paragraphs and print the text and bounding box coordinates
        for ((x1, y1, x2, y2), text) in boxes:
            #cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_text += f"Text:\\n{text}\\n\\n"
        print(output_text)
        return {"output_text" : output_text}
    else:
        return {"output_text" : None}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "image_path" : "string",
        },
        "outputs": {
            "output_text" : "string",
        }
    },
    "translate_language" : {
        "function_name": "translate_language",
        "import_string" : None,
        "function_string" : """
async def translate_language(node, input_text=None, input_language=None, output_language="English"):
    if input_text:
        if input_language != output_language:
            global cohere_api_key
            co = cohere.Client(cohere_api_key) # This is your trial API key
            fix_prompt = f'Translate each sentence into {output_language}, output only the translation:'
            response = co.generate(
                model='c4ai-aya-23',
                prompt=fix_prompt + input_text,
                max_tokens=20000,
                temperature=0.9,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE')
            print("Translated", response.generations[0].text)
            output_text = response.generations[0].text
            return {"output_text" : output_text}
        else:
            return {"output_text" : input_text}
    else:
        return {"output_text" : None}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "input_text" : "string",
            "input_language" : "string",
            "output_language" : "string"
         },
        "outputs": {
            "output_text" : "string",
        }
    },
    "detect_language" : {
        "function_name": "detect_language",
        "import_string" : None,
        "function_string" : """
async def detect_language(node, input_text=None):
    if input_text:
        try:
            co = cohere.Client(cohere_api_key) # This is your trial API key
            # Detect the language of the text
            # Print the language code and name
            response = co.generate(
            model='c4ai-aya-23',
            prompt="Input Text:\\n" + input_text + "\\nDetect language, output only ISO 639 language code in format, based on language not content:\\ncode: en,fr,jp,etc.",
            max_tokens=30,
            temperature=0.1,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE')
            language_code=response.generations[0].text
            print(response.generations[0].text)
            language = language_codes[language_code]
            print(language_code)
            print("language: ", language)
            return {"language" : language}
        except Exception as e:
            print(f"Error: {e}")
            return {"language" : "unknown"}
    else:
        return {"language" : "unknown"}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "input_text" : "string",
        },
        "outputs": {
            "language" : "string",
        }
    },
    "delay" : {
        "function_name": "delay",
        "import_string" : None,
        "function_string" : """
async def delay(node, delay_seconds=None):
    if node.trigger_in.startswith("time_delta_seconds"):
        
        asyncio.sleep(delay_seconds)
    elif delay_seconds:
        asyncio.sleep(delay_seconds)
    return None
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "delay_seconds": "string",
        },
        "outputs": {
        }
    },
    "time_delta_seconds_from_now" : {
        "function_name": "time_delta_seconds",
        "import_string" : None,
        "function_string" : """
async def time_delta_seconds(node, given_date_time_str):
    now = datetime.now()
    
    # Given date and time
    given_date_time = datetime.strptime(given_date_time_str, "%Y-%m-%d %H:%M:%S")
    
    # Calculate the difference
    delta = now - given_date_time
    
    # Convert the difference to seconds
    delta_seconds = delta.total_seconds()
    
    return {"seconds" : delta_seconds}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "given_date_time_str": "string",
        },
        "outputs": {
            "seconds" : "num",
        }
    },
    "decide_output_language" : {
        "function_name": "decide_output_language",
        "import_string" : None,
        "function_string" : """
async def decide_output_language(node, user_language=None, listener_language=None, user_prompt=None, user_info=None, listener_info=None):
    if input_text:
        try:
            language_code = detect(user_prompt)
            language = language_codes[language_code]
            return {"language" : language}
        except Exception as e:
            print(f"Error: {e}")
            return {"language" : "English"}
    else:
        return {"language" : "English"}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "user_prompt": "string",
            "user_language": "string",
            "user_info": "string",
            "listener_language": "string",
            "listener_info": "string"
        },
        "outputs": {
            "language" : "string",
        }
    },
    
}

#def newNode(node):

def generate_node_id(name):
    #UUID based
    #Time based
    # get the current time in seconds since the epoch
    seconds = time.time()
    return f"{name}_{seconds}"

async_node_string = None
async_nodes = {}
functions = {
    # Add more functions here as needed
}

    
def generate_node(name, pos = [0,0], input_addresses=None, output_args=None, trigger_out=None, node_id=None):
    regenerate = True
    print(node_id)
    if node_id == None:
        regenerate = False
        
        node_id = generate_node_id(name)
        print("Generating new", node_id)
    #print("printing node id: ", node_id, name, pos)
    
    nodes[node_id] = DraggableLabel(
        name = name,
        node_id = node_id,
        inputs = node_init[name]["inputs"],
        outputs = node_init[name]["outputs"],
        pos = pos)
    print(nodes[node_id])
    print(name, nodes[node_id], nodes[node_id].inputs, node_init[name])
    outputs = {}
    inputs = []
    t_out = []
    
    #print(inputs)
    #print(outputs)
    #print(t_out)
    
    async_nodes[node_id] = AsyncNode(name, input_addresses=inputs, output_args=outputs, trigger_out=t_out, node_id=node_id)
    #print(async_nodes[node_id])
    if not output_args == None:
        outputs = {}
        if node_init[name]["outputs"]:
            for i in node_init[name]["outputs"]:
                #print(i)
                outputs.update({i : None})
        for i in output_args:
            outputs.update({i : output_args[i]})
    
    if not input_addresses == None:
        inputs = input_addresses
    
    if not trigger_out == None:
        t_out = trigger_out
    
    #print(inputs)
    #print(outputs)
    #print(t_out)
        #{"node" : i["node"], "arg_name" : None, "target" : None}
    input_addresses_2 = copy.deepcopy(inputs)
    output_args_2 = copy.deepcopy(outputs)
    trigger_out_2 = copy.deepcopy(t_out)
    
    node_info[node_id] = {
        "name" : name,
        "node_id" : node_id,
        "input_addresses" : input_addresses_2,
        "output_args" : output_args_2,
        "pos" : pos,
        "trigger_out" : trigger_out_2,
    }
    
    print("Inputs", inputs, input_addresses)
    for i in inputs:
        print(i["node"])
        for j in async_nodes:
            print(j)
        
        if i["node"] not in async_nodes:
            async_nodes[i["node"]] = AsyncNode(name, input_addresses=inputs, output_args=outputs, trigger_out=t_out, node_id=node_id)
            
        print("someasync: ", async_nodes[i["node"]], node_id)
        async_nodes[node_id].input_addresses.append({"node" : async_nodes[i["node"]], "arg_name" : i["arg_name"], "target" : i["target"]})
    async_nodes[node_id].output_args = outputs
    for i in t_out:
        #print("t_out: ", i)
        async_nodes[node_id].trigger_out.append(i)
    
    """
    inputs = []
    for i in input_addresses:
        #print(i)
        inputs.update({i : None})
    for i in output_args:
        outputs.update({{'node': , 'arg_name': , 'target': }]})
    """
    
    return node_id
    
def generate_async_string():
    #Generate Async Node String
    #async_nodes[node_id] = AsyncNode(name, input_addresses=[{"node": source, "arg_name": "a", "target": "a"}, {"node": source, "arg_name": "b", "target" : "b"}], output_args={"a": 5})
    pass

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
    
#So like the LLM will retrieve nodes from files instead of just documents and codes, I shouldnt be documenting this.
def generate_node_with_llm(inputs, outputs, description, documentation, name):
    #Example:
    #"add", input_addresses=[{"node": source, "arg_name": "a", "target": "a"}, {"node": source, "arg_name": "b", "target" : "b"}], output_args={"a": 5}
    name = None
    import_string = None
    function_string = None
    inputs = []
    outputs = []
    description = None
    documentation = None
    #Fill the thing
    #Test LLM
    #Generate code first in such format
    #I do over planning on the top level and forgotten need to construct prompt format lol
    #Prompt Structure
    """
    Given inputs
    {inputs}
    {outputs}
    
    
    """
    #Put in node init
    node_init[name] = {
        "function_name": name,
        "import_string" : import_string,
        "function_string" : function_string,
        "description" : description,
        "documentation" : documentation,
        "inputs" : inputs,
        "outputs": outputs,
        }
    
    # Convert the dictionary to a JSON string
    node_init_json = json.dumps(node_init[name], indent=4)

    # Save the JSON string to a file
    file_path = f"{name}.json"
    with open(file_path, "w") as json_file:
        json_file.write(node_init_json)
    
    print(f"Node init data has been saved to {file_path}")
    #Save documentation separately
    
def add_node_from_init():
    pass
"""
print(node_init["add"]["inputs"])

for i in node_init["add"]["inputs"]:
    print(i, node_init["add"]["inputs"])
"""

KV = '''
<DraggableLabelScreen>:
    name: 'draggable_label_screen'
    
<RenderScreen>
    name: 'render_screen'
    
<NewNodeScreen>
    name: 'new_node_screen'

<SelectNodeScreen>
    name: 'select_node_screen'

<WidgetTreeScreen>
    name: 'widget_tree_screen'
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

<ChatboxScreen>:
    name: "chatbox"
    MDBoxLayout:
        orientation: 'vertical'
        padding: [0,0,0,100]
        BoxLayout:
            orientation: 'horizontal'
            size_hint: (1, None)
            height: 40
            pos_hint: {'top': 1}

            Button:
                text: 'Back'
                on_press: app.root.current = "draggable_label_screen"
            
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
                        text: "gemini_logo"
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
                        text: "camera_icon"
                        pos_hint: {'center_x': .5, 'center_y': 0.5}  # Center the MDIconButton initially
                        size_hint: None, None
                        size: 50, 50
                        padding_x: 10
                        #pos: self.parent.center_x - self.width / 2, self.parent.center_y - self.height / 2  # Position the MDIconButton at the center of its parent
                        
                        Image:
                            source: "images/camera_icon.png"
                            size_hint: None, None
                            size: 50, 50  # Set a fixed size for the Image
                            pos_hint: {'center_x': 0.5, 'center_y': 0.5}  # Center the Image initially
        BoxLayout:
            orientation: 'horizontal'
            size_hint: (1, None)
            height: 40
            canvas.before:
                Color:
                    rgba: 0.25, 0.25, 0.25, 1  # Background color
                Rectangle:
                    pos: self.pos
                    size: self.size
            MDIconButton:
                text: "file_attachment"
                pos_hint: {'center_x': .5, 'center_y': 0.5}  # Center the MDIconButton initially
                size_hint: None, None
                size: 40, 40
                padding_x: 10
                #pos: self.parent.center_x - self.width / 2, self.parent.center_y - self.height / 2  # Position the MDIconButton at the center of its parent
                #on_release: app.button_pressed()  # Define the action to be taken when the button is released
                Image:
                    source: "images/file_attach.png"
                    size_hint: None, None
                    size: 40, 40  # Set a fixed size for the Image
                    pos_hint: {'center_x': 0.5, 'center_y': 0.5}  # Center the Image initially
            MDIconButton:
                text: "other_obj"
                size_hint: None, None
                size: 40, 40
                pos_hint: {'right': 1, 'center_y': 0.5}  # Position the MDIconButton at the rightmost side, center vertically
                #pos: (self.parent.width - self.width - 10, self.parent.center_y - self.height / 2)  # Position the MDIconButton to the rightmost side
                padding_x: 10
                # on_release: app.button_pressed()  # Define the action to be taken when the button is released
                Image:
                    source: "images/file_attach.png"
                    size_hint: None, None
                    size: 40, 40  # Set a fixed size for the Image
                    pos_hint: {'center_x': 0.5, 'center_y': 0.5}  # Center the Image initially
     
ScreenManager:
    id: screen_manager
    DraggableLabelScreen:
    ChatboxScreen:
    SelectNodeScreen:
    RenderScreen:
    NewNodeScreen:
    WidgetTreeScreen:
'''

class CustomComponent(BoxLayout):
    pass

# Custom component with Label on the left and Button on the right
class NewNodeComponent(BoxLayout):
    def __init__(self, text, **kwargs):
        super(NewNodeComponent, self).__init__(orientation='horizontal', **kwargs)
        self.text = text
        
        self.label = Label(text=text, size_hint_x=0.7, halign='left', valign='middle')
        self.label.bind(size=self.label.setter('text_size'))  # Ensure the text size matches the label size
        
        self.button_box = BoxLayout(size_hint_x=0.3, orientation="horizontal")
        
        self.add_button = Button(text="Add")
        self.add_button.bind(on_press=self.button_on_press)
        
        self.edit_button = Button(text="Edit")
        
        self.button_box.add_widget(self.add_button)
        self.button_box.add_widget(self.edit_button)
        
        self.add_widget(self.label)
        self.add_widget(self.button_box)
        
        
    def button_on_press(self, instance):
        try:
            print(self.text)
            app = MDApp.get_running_app()
            app.root.get_screen('draggable_label_screen').new_node(node_name=self.text, new_node_id=None)
            #app.manager.transition = NoTransition()
            app.root.current = "draggable_label_screen"

        except Exception as e:
            print(e)

class SelectNodeScreen(Screen):
    def __init__(self, **kwargs):
        super(SelectNodeScreen, self).__init__(**kwargs)
        
        # Create the main layout for the screen
        screen_layout = BoxLayout(orientation='vertical')
        
        # Create the back button layout
        back_box = BoxLayout(size_hint=(1, None), height=40)
        back_button = Button(text="Back")
        back_button.bind(on_press=self.back_button_on_press)
        back_box.add_widget(back_button)
        
        # Create the main scroll view
        main_scroll = ScrollView(size_hint=(1, 1))
        
        # Create the main layout inside the scroll view
        self.main_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.main_layout.bind(minimum_height=self.main_layout.setter('height'))
        
        # Add custom components to the main layout
        for i in node_init:  # Adding multiple custom components
            custom_component = NewNodeComponent(text=f"{i}")
            custom_component.size_hint_y = None
            custom_component.height = 50
            self.main_layout.add_widget(custom_component)
        
        # Add the main layout to the scroll view
        main_scroll.add_widget(self.main_layout)
        
        # Add the back button and scroll view to the screen layout
        screen_layout.add_widget(back_box)
        screen_layout.add_widget(main_scroll)
        
        # Add the screen layout to the screen
        self.add_widget(screen_layout)
        
    def back_button_on_press(self, instance):
        app = MDApp.get_running_app()
        self.manager.transition = NoTransition()
        self.manager.current = 'draggable_label_screen'
        
class ChatboxScreen(Screen):
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


class MyTreeView(TreeView):
    def __init__(self, widget_tree_screen, **kwargs):
        super().__init__(**kwargs)
        self.widget_tree_screen = widget_tree_screen
        self.size_hint_y = None
        self.bind(minimum_height=self.setter('height'))
        self.bind(selected_node=self.on_node_selected)
    
    def clear_widget_tree(self):
        # Remove all nodes from the TreeView
        for node in self.root.nodes[:]:
            self.remove_node(node)
    
    def add_widget_tree(self, parent_node, tree):
        widget_name = str(tree['widget']['name'])
        if 'reference' in tree['widget'] and (isinstance(tree['widget']['reference'], Button) or isinstance(tree['widget']['reference'], MDIconButton)):
            widget_name += f" : {tree['widget']['reference'].text}"
        node = self.add_node(TreeViewLabel(text=widget_name), parent=parent_node)
        node.widget_ref = tree['widget'].get('reference', None)
        for child in tree['children']:
            self.add_widget_tree(node, child)

    def on_node_selected(self, tree_view, node):
        if node and hasattr(node, 'widget_ref'):
            widget_ref = node.widget_ref
            self.widget_tree_screen.display_widget_properties(widget_ref)
button_nodes = {}
class WidgetTreeScreen(Screen):
    def __init__(self, widget_tree=None, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.selected_node = None
        self.selected_widget = None
        
        self.top_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, pos_hint={'top': 1})
        self.back_button = Button(text='Back', on_press=self.switch_to_screen)
        self.top_layout.add_widget(self.back_button)
        
        # Top box with scrollable tree view
        self.top_box = BoxLayout(orientation='vertical', size_hint_y=0.7)
        self.scroll_view = ScrollView()
        self.tree_view = MyTreeView(widget_tree_screen=self, root_options=dict(text='Widget Tree'), hide_root=False)
        self.scroll_view.add_widget(self.tree_view)
        self.top_box.add_widget(self.scroll_view)
        
        # Bottom box with scrollable TextInput
        self.bottom_box = BoxLayout(orientation='vertical', size_hint_y=0.3)
        self.button_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        self.property_label = Label(text="Properties", size_hint_x=0.75)
        self.create_node_label = Button(text="Create Node", size_hint_x=.25, on_press=self.create_node)
        self.text_input = TextInput(readonly=True, multiline=True, size_hint_y=None)
        self.text_scroll = ScrollView()
        self.bottom_box.add_widget(self.button_box)
        self.text_scroll.add_widget(self.text_input)
        self.bottom_box.add_widget(self.text_scroll)
        
        self.button_box.add_widget(self.property_label)
        self.button_box.add_widget(self.create_node_label)
        
        # Add both boxes to the main layout
        self.layout.add_widget(self.top_layout)
        self.layout.add_widget(self.top_box)
        self.layout.add_widget(self.bottom_box)
        self.add_widget(self.layout)

        if widget_tree:
            self.update_widget_tree(widget_tree)
    def switch_to_screen(self, instance):
        # Switch to 'chatbox'
        self.manager.transition = NoTransition()
        self.manager.current = 'draggable_label_screen'
    
    
    def update_widget_tree(self, widget_tree):
        #self.tree_view.clear_widgets()
        self.tree_view.add_widget_tree(None, widget_tree)

    def create_node(self, instance):
        if isinstance(self.selected_widget, Button) or isinstance(self.selected_widget, MDIconButton) and self.selected_node not in button_nodes:
            button_nodes[self.selected_node] = self.selected_widget
            print(self.selected_widget, self.selected_node)
            # Bind the button to a lambda function with a variable
            self.selected_widget.bind(on_press=lambda instance, node=self.selected_node: self.on_run_press_wrapper(instance, node))
            print("Bound Node")
            
            node_init[self.selected_node] = {
                "function_name": self.selected_node,
                "import_string" : None,
                "function_string" : None,
                "description" : None,
                "documentation" : None,
                "inputs" : {
                    
                },
                "outputs": {
                }
            }

            #ui_node_screens[self.selected_node] = 
            print(node_init[self.selected_node])
            self.parent.get_screen('draggable_label_screen').new_node(node_name=self.selected_node, new_node_id=self.selected_node)
            print(nodes[self.selected_node])
            self.manager.transition = NoTransition()
            self.manager.current = 'draggable_label_screen'
    async def on_run_press(self, node):
        #print("Run Pressed")
        # Search for ignition nodes and trigger them once.
        tasks = []
        print("Running: ", node)
        for i in node_info:
            if node_info[i]["name"] == node:
                #print(i, async_nodes[i])
                try:
                    # Your existing code here...
                    print("someasync: ", async_nodes[i].trigger_out, i)
                    tasks.append(asyncio.create_task(async_nodes[i].trigger()))
                    
                    # Your existing code here...
                except RecursionError:
                    print("Maximum recursion depth reached. Stopping program.")
                    # Additional cleanup or handling here if needed
        await asyncio.gather(*tasks)
        
    def on_run_press_wrapper(self, instance, node):
        def run_coroutine_in_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.on_run_press(node))

        threading.Thread(target=run_coroutine_in_event_loop).start()    
    def parse_to_valid_function_name(self, input_string):
        # Replace invalid characters with underscores
        valid_name = re.sub(r'[^0-9a-zA-Z_]', '_', input_string)

        # Ensure the name starts with a letter or underscore
        if not valid_name[0].isalpha() and valid_name[0] != '_':
            valid_name = '_' + valid_name

        return valid_name
    def display_widget_properties(self, widget):
        self.text_input.text = ""
        print(widget)
        for attr_name in dir(widget):
            if not attr_name.startswith('_'):
                try:
                    value = getattr(widget, attr_name)
                    if not callable(value):
                        self.text_input.text += f"{attr_name}: {value}\n"
                        if attr_name == "text":
                            self.selected_node = f"Button : {value}"
                            #self.selected_node = self.parse_to_valid_function_name(self.selected_node)
                            self.selected_widget = widget
                            #Create a node_init of name
                            #Generate bindings
                            
                            pass
                except Exception as e:
                    self.text_input.text += f"{attr_name}: Could not retrieve value ({e})\n"
        self.text_input.height = max(200, self.text_input.minimum_height)

class DraggableLabelScreen(Screen):
    def __init__(self, **kwargs):
        super(DraggableLabelScreen, self).__init__(**kwargs)
        import_string = """
import asyncio

print("imported")
        """
        exec(import_string, globals())
        function_string = """
async def ignition(node):
    print("Ignition")
    return None

print("Updated function string")
        """
        exec(function_string, globals())



        for i in node_init:
            function_name = node_init[i]["function_name"]
            formatted_string = f"""
functions.update({{\"{function_name}\": {function_name}}})
print(\"Added\", {function_name})
            """
            try:
                exec(node_init[i]["import_string"], globals())
            except:
                pass
            exec(node_init[i]["function_string"], globals())
            exec(formatted_string, globals())
        self.layout = BoxLayout(orientation='vertical')
        self.build()
        
    def build(self):
        root = FloatLayout()
        
        mouse_widget = MousePositionWidget(size_hint_y=None, height=40)
        self.layout.add_widget(mouse_widget)
        #Reset nodes
        """
        generate_node("ignition", pos = [50, 400])
        
        generate_node("select_model", pos = [50, 300])
        generate_node("user_input", pos = [50, 200])
        generate_node("context", pos = [50, 100])
        
        
        generate_node("prompt", pos = [300, 100])
        
        generate_node("display_output", pos = [300, 500])
        """
        #generate_node("prompt", pos = [300, 150])
        
        print("printing nodes")
        for i in nodes:
            print(i)
            self.layout.add_widget(nodes[i])
        
        for i in nodes:
            print(i)
            nodes[i].pos = (node_info[i]["pos"][0], node_info[i]["pos"][1])
        #layout.add_widget(nodes_layout)

        root.add_widget(self.layout)
        
        # Floating layout
        top_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, pos_hint={'top': 1})
        back_button = Button(text='Chatbot')
        top_layout.add_widget(back_button)
        render_button = Button(text='Renderer', on_press=self.switch_to_renderer)
        top_layout.add_widget(render_button)
        # Bind button press to switch_to_screen method
        back_button.bind(on_press=self.switch_to_screen)
        
        root.add_widget(top_layout)
        # Floating layout
        floating_layout = BoxLayout(orientation='horizontal', size_hint=(1, .05), pos=(0, 0))
        
        run_code = Button(text='Run Code')
        floating_layout.add_widget(run_code)
        run_code.bind(on_press=self.on_run_press_wrapper)
        
        add_node = Button(text='Add Node')
        floating_layout.add_widget(add_node)
        add_node.bind(on_press=self.add_node_on_press)
        
        new_node_button = Button(text='New Node')
        floating_layout.add_widget(new_node_button)
        new_node_button.bind(on_press=self.new_node_on_press)
        
        object_explorer_button = Button(text='Object Explorer')
        floating_layout.add_widget(object_explorer_button)
        object_explorer_button.bind(on_press=self.object_explorer_on_press)
        
        save_nodes = Button(text='Save Nodes')
        floating_layout.add_widget(save_nodes)
        save_nodes.bind(on_press=self.save_nodes)
        
        load_nodes = Button(text='Load Nodes')
        floating_layout.add_widget(load_nodes)
        load_nodes.bind(on_press=self.load_nodes)
        
        #floating_layout.add_widget(run_code)
        root.add_widget(floating_layout)

        self.add_widget(root)
        
    def generate_widget_tree(self, widget, level=0):
        tree = {
            'widget': {
                'name': widget.__class__.__name__,
                'properties': {},
                'reference': widget
            },
            'children': []
        }
        for attr_name in dir(widget):
            if not attr_name.startswith('_'):
                try:
                    value = getattr(widget, attr_name)
                    if not callable(value):
                        tree['widget']['properties'][attr_name] = value
                except Exception as e:
                    tree['widget']['properties'][attr_name] = f"Could not retrieve value ({e})"

        for child in widget.children:
            tree['children'].append(self.generate_widget_tree(child, level + 1))

        return tree


    def print_widget_tree(self, tree, level=0):
        indent = '  ' * level
        print(f"{indent}{tree['widget']['name']}")
        for key, value in tree['widget']['properties'].items():
            print(f"{indent}  {key}: {value}")
        for child in tree['children']:
            self.print_widget_tree(child, level + 1)

    def object_explorer_on_press(self, instance):
        """
        for screen in self.parent.screens:
            widget_tree = self.generate_widget_tree(screen)
            self.print_widget_tree(widget_tree)
        """
        
        screen = self.parent
        self.parent.get_screen('widget_tree_screen').tree_view.clear_widget_tree()
        for screen2 in self.parent.screens:
            print(screen2)
            widget_tree = self.generate_widget_tree(screen2)
            self.parent.get_screen('widget_tree_screen').update_widget_tree(widget_tree)
        #self.print_widget_tree(widget_tree)
        
        #print(widget_tree)
        self.manager.transition = NoTransition()
        
        self.manager.current = 'widget_tree_screen'
        
    def new_node_on_press(self, instance):
        app = MDApp.get_running_app()
        self.manager.transition = NoTransition()
        self.manager.current = 'new_node_screen'
        
    def add_node_on_press(self, instance):
        #app = MDApp.get_running_app()
        self.manager.transition = NoTransition()
        self.manager.current = 'select_node_screen'
        
    def save_nodes(self, instance):
        #Save line points instead
        print("Line: ", lines)
        saved_lines = {}
        for i in lines:
            saved_lines.update({i: lines[i].points})
            print(saved_lines[i])
        f = open("saved_lines.json", "w")
        f.write(json.dumps(saved_lines))
        f.close()
        
        print("Connections: ", connections)
        f = open("connections.json", "w")
        f.write(json.dumps(connections))
        f.close()
        #print("Nodes: ", nodes)

        print("Node Info: ", node_info)
        print("Async Nodes: ", async_nodes)
        f = open("node_info.json", "w")
        f.write(json.dumps(node_info))
        f.close()
        
        #Save node_init too
        print("Node Init: ")
        f = open("node_init.json", "w")
        f.write(json.dumps(node_init))
        f.close()
        #Save UI component screen references, but first create separate reference, especially which screen it is
        
    def load_nodes(self, instance):
        #Load lines
        
        f = open("saved_lines.json", "r")
        saved_lines = json.load(f)
        global lines
        #lines = {}
        for i in saved_lines:
            with self.canvas:
                Color(1, 0, 0)
                lines[i] = (Line(points=saved_lines[i]))
                lines_canvas[i] = self.canvas
            #print(saved_lines[i])
        print("Line: ", lines)
        
        global connections
        #Load connections
        #connections = {}
        f = open("connections.json", "r")
        connections = json.load(f)
        print("Connections: ", connections)
        

        #Load node_init

        global node_init
        f = open("node_init.json", "r")
        node_init_temp = json.load(f)
        for i in node_init_temp:
            if not i in node_init:
                node_init[i] = copy.deepcopy(node_init_temp[i])
        
        global node_info
        #Load node infos
        #node_info = {}
        #nodes = {}
        f = open("node_info.json", "r")
        node_info_temp = json.load(f)
        #print("Connections: ", node_info)
        #print("Node Info: ", node_info)
        
        #Generate Nodes with node infos

        
        print("Node Info: ", node_info_temp)
        for i in node_info_temp:
            print(i)
        for i in node_info_temp:
            #(name, pos = [0,0], input_addresses=[], output_args={}, trigger_out=[], node_id=None)
            print(i)
            #print(node_info[i]["name"])
            #print(node_info[i]["pos"])
            generate_node(name=node_info_temp[i]["name"], pos=node_info_temp[i]["pos"], input_addresses=node_info_temp[i]["input_addresses"], output_args=node_info_temp[i]["output_args"], trigger_out=node_info_temp[i]["trigger_out"], node_id=node_info_temp[i]["node_id"])
            node_info[i] = copy.deepcopy(node_info_temp[i])
            #Rebind those starting in Button or MDButton with tree view wrapper with node name variable
            #But first find the reference, iterate through widget_tree and find element of that name
            #If starts with Button : get the screen reference and find the element there
            
            if i.startswith("Button : "):
                for screen in self.parent.screens:
                    button_text = i.replace("Button : ", "")
                    for widget in screen.walk():
                        if (isinstance(widget, Button) or isinstance(widget, MDIconButton)) and widget.text == button_text:
                            print(f"Found button with text '{button_text}' in ScreenTwo, reference: {widget}")
                            widget.bind(on_press=lambda instance, node=i: screen.on_run_press_wrapper(instance, node))
                            break
            
        for i in nodes:
            try:
                self.layout.add_widget(nodes[i])
            except:
                pass
        print("Nodes: ", nodes)
        #Update trigger_connections located in AsyncNode trigger_out
        for i in async_nodes:
            async_nodes[i].trigger_out = []
            for j in node_info[i]["trigger_out"]:
                async_nodes[i].trigger_out.append(async_nodes[j])
            #print(node_info[i]["trigger_out"])
            #print(async_nodes[i].trigger_out)
            
    def new_node(self, node_name, new_node_id):
        node_id = generate_node(node_name, pos = [100, 200], node_id=new_node_id)
        self.layout.add_widget(nodes[node_id])

        global added_node
        global nodes_regenerated
        added_node = True
        nodes_regenerated = 0
        """
        for i in nodes:
            self.layout.remove_widget(nodes[i])
            #self.layout.add_widget(nodes[i])
        """
        
        for i in nodes:
            print(i, nodes[i].pos)
            node_info[i]["pos"] = (nodes[i].pos[0], nodes[i].pos[1])
            nodes[i].regenerated = True
        
            #generate_node(node_info[i]["name"], pos = [nodes[i].pos[0], nodes[i].pos[1]], node_id=i)
    
    async def on_run_press(self):
        #print("Run Pressed")
        # Search for ignition nodes and trigger them once.
        tasks = []
        
        for i in node_info:
            if node_info[i]["name"] == "ignition":
                #print(i, async_nodes[i])
                try:
                    # Your existing code here...
                    print("someasync: ", async_nodes[i].trigger_out, i)
                    tasks.append(asyncio.create_task(async_nodes[i].trigger()))
                    
                    # Your existing code here...
                except RecursionError:
                    print("Maximum recursion depth reached. Stopping program.")
                    # Additional cleanup or handling here if needed
        await asyncio.gather(*tasks)
        
    def on_run_press_wrapper(self, instance):
        def run_coroutine_in_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.on_run_press())

        threading.Thread(target=run_coroutine_in_event_loop).start()

    def update_ui(self):
        # Update your UI here
        print("test")
        pass
    
    def switch_to_screen(self, instance):
        # Switch to 'chatbox'
        self.manager.transition = NoTransition()
        self.manager.current = 'chatbox'
        
    def switch_to_renderer(self, instance):
        # Switch to 'chatbox'
        self.manager.transition = NoTransition()
        self.manager.current = 'render_screen'
class DraggableLabelApp(MDApp):
    past_messages = []
    def build(self):
        self.theme_cls.theme_style = 'Dark'
        return Builder.load_string(KV)
    
    def switch_screen(self, screen_name):
        screen_manager = self.root.ids.screen_manager
        screen_manager.transition = NoTransition()
        screen_manager.current = screen_name
        
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
        
    # Function to continue the conversation
    def continue_conversation(self, model=None, context=None):
        #print(past_messages)
        # Create the chat completion request with updated past messages
        if context:
            self.add_message("context", context)
        chat_completion = client.chat.completions.create(
          messages=self.past_messages,
          model=model or "mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        print(self.past_messages)
        response = chat_completion.choices[0].message.content
        # Update the past messages list with the new chat completion
        response = response.replace("\\_", "_")
        self.add_message("assistant", response)

        # Print the assistant's response
        print("Bot: ", response)
        return response
    """
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
    """
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
        
        generate_code = f"User Input: {input_text}\nInstruct Types:\n0: Normal, normal conversation\n1: Generate Image, if user wants to generate an image\n2: Search Facebook, if user wants to search Facebook.\n3: Search Google, If user wants to do Web Search or if you don't know the answer or wants updated answer.4: Search Google with Images, If user wants to search images of an object\n, output only the number of the instruct type, with format: \nFormat: instruct type:<number>"
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
            
        generated_image_path = self.generate_image(image_prompt)
        self.add_message("system", f"System: You've successfully generated an image for the user, as you are connected to an image generating AI, the generated image prompt: {image_prompt}, you will just say to the user that here's the image, and describe the image.")
        return generated_image_path
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
        
        filename = f"images/generated_image_{time.time()}.jpeg"
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception(str(response.json()))
        
        return filename
    
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
            self.root.get_screen("chatbox").ids.output_text.text = str(e)
            return
        finally:
            # Restore the original standard output
            sys.stdout = sys_stdout
        print(captured_output.get_value())
        return captured_output.get_value()
        
    async def on_run_press(self):
        #print("Run Pressed")
        # Search for ignition nodes and trigger them once.
        for i in node_info:
            if node_info[i]["name"] == "ignition":
                #print(i, async_nodes[i])
                try:
                    # Your existing code here...
                    print("someasync: ", async_nodes[i].trigger_out, i)
                    await async_nodes[i].trigger()
                    
                    # Your existing code here...
                except RecursionError:
                    print("Maximum recursion depth reached. Stopping program.")
                    # Additional cleanup or handling here if needed
    
    def on_run_press_wrapper(self):
        def run_coroutine_in_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.on_run_press())

        threading.Thread(target=run_coroutine_in_event_loop).start()        
    def button_pressed(self):
        text_input = self.root.get_screen("chatbox").ids.text_input
        
        user_text = text_input.text
        
        #self.generate_documentation("somefibo.py")
        use_model = "together"
        if use_model == "gemini":
            """
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
            """
            grid_layout = self.root.get_screen("chatbox").ids.grid_layout
            grid_layout.add_widget(user_custom_component)
            grid_layout.add_widget(gemini_custom_component)
        if use_model == "together":
            # Add a new user message
            self.add_message("user", user_text)
            # Search documents with cohere rerank
            # From web
            # From file
            # In coherererank code
            
            #Send thing to user_input but first we gotta find the address of the userinput, we can also pass a custom node_id or modify it
            for i in node_info:
                #print(i)
                if node_info[i]["name"] == "user_input":
                    print(i)
                    async_nodes[i].output_args.update({"user_input" : user_text})
                    print(async_nodes[i].output_args)
            
            
            self.on_run_press_wrapper()
            #display_output_2(user_text)
            
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
            """
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
            
            grid_layout = self.root.get_screen("chatbox").ids.grid_layout
            grid_layout.add_widget(user_custom_component)
            #grid_layout.add_widget(CustomImageComponent(img_source="images/bug.png"))
            grid_layout.add_widget(bot_custom_component)
            if instruct_type == 1:
                grid_layout.add_widget(CustomImageComponent(img_source="images/generated_image.jpeg"))
            """
            
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
    """
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
    """
if __name__ == '__main__':
    DraggableLabelApp().run()
    