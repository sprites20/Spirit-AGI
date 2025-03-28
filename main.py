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
from kivymd.uix.button import MDIconButton

from kivy.uix.behaviors import DragBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.treeview import TreeView, TreeViewLabel
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.codeinput import CodeInput

from kivy.resources import resource_find

from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST, glCullFace, GL_BACK
from kivy.graphics import RenderContext, Callback, PushMatrix, PopMatrix, \
    Color, Translate, Rotate, Mesh, UpdateNormalMatrix, BindTexture, Rectangle, Ellipse, Line
from kivy.graphics.texture import Texture

from kivy.properties import Property
from kivy.properties import NumericProperty

from objloader import ObjFile
import tkinter as tk
from tkinter import filedialog
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.properties import NumericProperty
from kivy.core.audio import SoundLoader
import pyttsx3

from kivy.core.window import Window

from kivy.app import App
import math
import os
import requests
import asyncio
import shutil
from functools import partial
#from pyppeteer import launch
import base64
from geopy.geocoders import Nominatim

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import duckdb
from pyproj import Proj, transform

from pygments.lexers import PythonLexer

from textblob import TextBlob

from openai import OpenAI

from io import StringIO
from pathlib import Path

from datetime import datetime, timedelta

#import google.generativeai as genai
from io import StringIO
import subprocess

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
from pathlib import Path
import traceback

import pybullet as p
import pybullet_data

from pyppeteer import launch
from urllib.parse import quote, urlparse

import pytesseract
import cv2

import ast

#from mistralai.client import MistralClient
#from mistralai.models.chat_completion import ChatMessage

import cohere

import nest_asyncio

from huggingface_hub import HfApi, login, upload_file
import pandas as pd
import h5py
from datasets import Dataset, DatasetDict

nest_asyncio.apply()  # Apply the nest_asyncio patch

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

TOGETHER_API_KEY = "4070baa3baed3400f79377ea3b4221f2024725f970bbf02dd0b6d4fba2175bc6"

client = OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1',
)

client_for_instruct = OpenAI(
      api_key=TOGETHER_API_KEY,
      base_url='https://api.together.xyz/v1',
)

"""
client = OpenAI(
    api_key="95375458de2544cfb665366052759dbb",
    base_url="https://api.aimlapi.com",
)
"""

"""
MISTRAL_API_KEY = "VuNE7EzbFp5QA0zoYl0LokvrTitF7yrg"
client_mistral = MistralClient(api_key=MISTRAL_API_KEY)
"""
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
            "function_name": self.name_input.text,
            "import_string" : None,
            "function_string" : self.current_code or self.description_textinput.text,
            "description" : self.current_description or self.description_textinput.text,
            "documentation" : self.current_documentation or self.description_textinput.text,
            "inputs" : ast.literal_eval(f"{{{self.input_textinput.text}}}"),
            "outputs": ast.literal_eval(f"{{{self.output_textinput.text}}}")
        }
        
        global node_init
        node_init[self.name_input.text] = data
        print(node_init)
        #Save node_init
        print("Node Init: ")
        # Ensure the 'nodes' directory exists
        os.makedirs("nodes", exist_ok=True)
        f = open(f"nodes/{self.name_input.text}.json", "w")
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
        
        gen_message = f"Generate python code given parameters.\n\nInputs:\n{inputs}\n\nOutputs:\n{outputs}\n\n{gen_from}:\n{context}\n\nIn this format: \n{doc_format}\n\nMake sure to replace inputs and outputs with the given names. Output nothing else, even after the code but the code with indentations. Enclose with ```python"
        new_message = ChatMessage(role = "user", content = gen_message)
        message.append(new_message)
        
        
        """
        model = "codestral-latest"
        chat_completion = client_mistral.chat(
            model=model,
            messages=message
        )
        """
        
        chat_completion = client.chat.completions.create(
          messages=message,
          model="o1-preview"
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
User Cases:
        """
        
        gen_message = f"Generate short description code given parameters.\n\nInputs:\n{inputs}\n\nOutputs:\n{outputs}\n\n{gen_from}:\n{context}\n\nIn this format: \n{doc_format}\n\n. Output nothing else but the description."
        new_message = ChatMessage(role = "user", content = gen_message)
        message.append(new_message)
        
        """
        model = "codestral-latest"
        chat_completion = client_mistral.chat(
            model=model,
            messages=message
        )
        """
        
        chat_completion = client.chat.completions.create(
          messages=message,
          model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
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
        model = "codestral-latest"
        chat_completion = client_mistral.chat(
            model=model,
            messages=message
        )
        """
        
        chat_completion = client.chat.completions.create(
          messages=message,
          model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
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

    def change_to_red(self):
        nodes[self.node_id].label_color.rgba = (1,0,0,1)
        
    def change_to_gray(self):
        nodes[self.node_id].label_color.rgba = (0.5, 0.5, 0.5, 1)

class AsyncNode:
    def __init__(self, function_name=None, node_id=None, input_addresses=[], output_args=None, trigger_out=[]):
        self.trigger_in = None
        self.trigger_out = []
        self.function_name = function_name
        self.input_addresses = input_addresses
        self.input_addresses_is_done = []
        self.trigger_out_taken = []
        self.output_args = {}
        self.node_id = node_id
        self.stop = {}
        self.args = {}
        self.trigger_out_callbacks = 0
        self.callbacks = {}
        self.isWaiting = threading.Event()  # Use threading.Event
        self.function_to_call = functions.get(self.function_name)

    def change_to_red(self):
        nodes[self.node_id].label_color.rgba = (1, 0, 0, 1)

    def change_to_gray(self):
        nodes[self.node_id].label_color.rgba = (0.5, 0.5, 0.5, 1)

    def trigger_node(self, node=None, user_id=None):
        node.trigger_in = self.node_id
        thread = threading.Thread(target=node.trigger, args=(node, user_id))
        thread.start()
        
    def callback(self, user_id):
        if user_id not in self.callbacks:
            self.callbacks[user_id] = False
        else:
            print("Callback: ", self.node_id)
            self.callbacks[user_id] = True

    def trigger(self, node=None, node_id=None, callback=None, user_id=None):
        # Change color to red
        Clock.schedule_once(lambda dt: self.change_to_red(), 0)
        
        user_id = user_id or "user"
        
        if user_id not in self.output_args:
            self.output_args[user_id] = {}
        
        try:
            if user_id not in self.stop:
                #print("Not in stop")
                self.stop[user_id] = False
        except Exception as e:
            print(e, "\n", user_id)
        #print(user_id, self.stop[user_id])
        if self.trigger_in:
            if self.trigger_in.startswith("stop_after"):
                self.stop[user_id] = True
            elif self.trigger_in.startswith("reset_outputs"):
                self.output_args[user_id].clear()  # Clear outputs for reset
                return None
                    
        # Get input arguments
        input_args = {}
        for address in self.input_addresses:
            node = address.get("node")
            arg_name = address.get("arg_name")
            target = address.get("target")
            try:
                input_args[target] = node.output_args[user_id].get(arg_name)
            except Exception as error:
                if user_id not in node.output_args:
                    node.output_args[user_id] = {}
                input_args[target] = None
                print("Error: ", error)
                print(node, arg_name, target, user_id, error)
                
                # Print the traceback for more details on where the error occurred
                traceback.print_exc()
        if self.function_to_call:
            function_done = threading.Event()  # Event to signal function completion
            #function_done.clear()  # Clear the event before running the function
            
            # Create a separate thread for the function call
            def run_function():
                output_args = self.function_to_call(self, **input_args) or {}
                self.output_args[user_id] = output_args
                
                # Change color to gray after processing
                Clock.schedule_once(lambda dt: self.change_to_gray(), 0)
                # Wait for the function to finish before continuing
                # Wait for the function to finish before continuing
                function_done.set()  # Signal that the function is done
                function_done.wait()  # Block until the function is complete

            thread = threading.Thread(target=run_function)
            thread.start()

            function_done.wait()  # Block until the function is complete
        
        for address in self.input_addresses:
            node = address.get("node")
            arg_name = address.get("arg_name")
            target = address.get("target")
            if arg_name == "user_id":
                try:
                    user_id = node.output_args[user_id].get(arg_name) or "user"
                    break
                except:
                    user_id = "user"
                    pass
        try:
            if user_id not in self.stop:
                #print(f"User Id {user_id} not in stop")
                self.stop[user_id] = False
        except Exception as e:
            print(e, "\n", user_id)
        
        if not self.stop[user_id]:
            #print(user_id, "Stopped?",self.stop[user_id])
            for node_to_run in self.trigger_out:
                self.trigger_node(node=node_to_run, user_id=user_id)
        self.stop[user_id] = False  # Reset the stop condition if needed
    """
    async def trigger(self, node=None, node_id = None, callback=None, user_id=None):
        user_id = user_id or "user"
        if user_id not in self.output_args:
            self.output_args[user_id] = {}
        if self.trigger_in:
            if self.trigger_in.startswith("stop_after"):
                self.stop = True
            elif self.trigger_in.startswith("reset_outputs"):
                print(f"Resetting output {self.node_id}")
                try:
                    for arg_name in self.output_args:
                        self.output_args[user_id][arg_name] = None
                except:
                    pass
                return None
        # Get the function from the dictionary based on the function_name
        
        #print(self, self.node_id)
        
        if self.function_to_call:
            #print(f"Calling function {self.function_name}")
            # Fetch input_args from input_addresses
            input_args = {}
            
            #print(self.input_addresses)
            #First we take the user_id
            Clock.schedule_once(lambda dt: self.change_to_red(), 0)
            
            for address in self.input_addresses:
                node = address.get("node")
                arg_name = address.get("arg_name")
                target = address.get("target")
                try:
                    input_args[target] = node.output_args[user_id].get(arg_name)
                    #Initiate callback
                    #print(node, arg_name, node.output_args[user_id].get(arg_name))
                except Exception as e:
                    print(node, arg_name, target, e)
            #Initiate callback
            
            
            #print(input_args[target])
            #Here replace thing in output args with whatever queued. If none use same thing
            #print("Input Addresses: ", self.input_addresses)
            #print("Input Args", input_args)
            # Pass input_args and self to the function
            # Schedule UI update in the main Kivy thread
            
            # Use asyncio.create_task() to avoid conflicts
            task = asyncio.create_task(self.function_to_call(self, **input_args))
            
            try:
                #print("Callback: ", node_id)
                await async_nodes[node_id].callback(user_id)
            except:
                pass
            
            output_args = await task
            Clock.schedule_once(lambda dt: self.change_to_gray(), 0)
            #print("Output args: ", output_args)
            
            # Update output_args with the function's output, appending new args and replacing existing ones
            #await self.isWaiting.wait()  # Wait until the event is set (instead of while-loop)
			#self.isWaiting.clear()  # Clear the event before setting it again
			#There may be other race conditions where 2 of the same nodes are waiting
            
            if user_id in self.callbacks:
                while not self.callbacks[user_id]:
                    pass
            
            try:
                for arg_name, value in output_args.items():
                    if arg_name not in self.output_args[user_id]:
                        self.output_args[user_id][arg_name] = value
            except:
                pass
            #print("Ran: ", self.node_id)
            #self.callbacks[user_id] = False
			#self.isWaiting.set()  # Indicate that the processing is complete
			
        #print(node)
        #print("Output args: ", self.output_args)
        #print(self.output_args)
        
        for node in self.trigger_out:
            #print(f"Triggering output node {node.function_name}")
            await node.trigger()
        
        # Limit concurrent execution using semaphore
        async with self.semaphore:
            if self.function_to_call:
                task = asyncio.create_task(self.function_to_call(self, **input_args))
                output_args = await task or {}
                
                # Clear previous user output args
                self.output_args[user_id] = {}
                
                # Process output arguments
                for arg_name, value in output_args.items():
                    self.output_args[user_id][arg_name] = value

        if not self.stop:
            await asyncio.gather(*(self.trigger_node(node=node_to_run, user_id=user_id) for node_to_run in self.trigger_out))

        self.stop = False  # Reset the stop condition if needed
    """
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
        self.bottom_rect_size = (self.size_x, 20 + 20 * max(len(self.outputs), len(self.inputs)))
        
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
            
            self.input_inputs = {}
            
            self.output_inputs = {}
            
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
        
        self.box_rect.pos = (self.offsetted_pos[0], self.offsetted_pos[1] - (20 + 20 * max(len(self.outputs), len(self.inputs))))
        self.box_rect.size = (self.width, 20 + 20 * max(len(self.outputs), len(self.inputs)))
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
                #Use del
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
            
            self.box_rect.pos = (temp_pos[0], temp_pos[1] - (20 + 20 * max(len(self.outputs), len(self.inputs))))
            self.box_rect.size = (self.width, 20 + 20 * max(len(self.outputs), len(self.inputs)))
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
            
            self.box_rect.pos = (self.x, self.y - (20 + 20 * max(len(self.outputs), len(self.inputs))))
            self.box_rect.size = (self.width, (20 + 20 * max(len(self.outputs), len(self.inputs))))
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
            
            self.box_rect.pos = (self.x, self.y - (20 + 20 * max(len(self.outputs), len(self.inputs))))
            self.box_rect.size = (self.width, (20 + 20 * max(len(self.outputs), len(self.inputs))))
            
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


def save_dicts_as_json_files(data, output_directory):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for key, value in data.items():
        file_path = output_path / f"{key}.json"
        with file_path.open('w') as f:
            json.dump(value, f, indent=4)


from flask import Flask, request, jsonify, send_from_directory, make_response, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS  # Import CORS
import os
import threading
from collections import defaultdict
import queue
from pyngrok import ngrok  # Import pyngrok

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://sprites20.github.io"]}})  # Restrict to your domain
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Set a folder to store the uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

clients = {}
user_queues = queue.Queue()
user_locks = threading.Lock()
public_url = None
conversations = {}  # Store conversations for users

# Queue for connection events
connection_queue = queue.Queue()
connection_lock = threading.Lock()

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    user_id = request.args.get('user_id')
    print(user_id)
    #Should update the past messages by retrieving conversation ids, but first we should create if non exists
    with connection_lock:
        # Add connection request to the connection queue
        connection_queue.put((user_id, sid))

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    user_id = None
    app = MDApp.get_running_app()
    for uid, stored_sid in clients.items():
        if stored_sid == sid:
            user_id = uid
            break
    if user_id:
        del clients[user_id]
        del app.past_messages[user_id]
        print(f"Client {user_id} disconnected")

@socketio.on('client_event')
def handle_client_event(data):
    user_id = data.get('sender')
    message = data.get('text')
    image_url = data.get('image')
    conversation_id = data.get('conversation_id')
    
    splitted = user_id.split("_")
    timestamp = splitted[0]
    user_id = splitted[1]# + "_" + splitted[2]
    
    # Create a response data structure
    response_data = {
        'text': message,
        'image': image_url,
        'sender': user_id,
        'conversation_id': conversation_id,
    }
    # Save the message in the user's conversation
    with user_locks:
        if user_id not in conversations:
            conversations[user_id] = {}
        
        if conversation_id not in conversations[user_id]:
            conversations[user_id][conversation_id] = []
        
        conversations[user_id][conversation_id].append(response_data)  # Store the message
        
    with user_locks:
        # Put the response data into the user's queue for processing
        user_queues.put(response_data)
        print(f"Received message from {user_id}: {message}")

# HTTP route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    user_id = request.form.get('user_id')

    # Check if the user ID is provided
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Create the user's directory if it doesn't exist
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'])
        os.makedirs(user_folder, exist_ok=True)

        # Save the file in the user's folder
        file_path = os.path.join(user_folder, file.filename)
        file.save(file_path)
        print(f"File {file.filename} uploaded by {user_id}")
        global public_url
        # Use regex to extract the Ngrok public URL and the local path
        match = re.search(r'"(https?://[^\s]+)" -> "(http://localhost[^\s]+)"', str(public_url))
        ngrok_public_url = None
        if match:
            ngrok_public_url = match.group(1)  # Extracts the Ngrok public URL
        
        # Construct the URL for the uploaded file
        file_url = f"{ngrok_public_url}/uploads/{file.filename}"

        # Return a success message with the file URL
        return jsonify({"message": "File uploaded successfully", "file_url": file_url}), 200
    
# Serve uploaded images
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    if filename.endswith(".mp4"):
        filename = os.path.join("uploads", filename)
        return serve_mp4_with_range(filename)
    else:
        response = make_response(send_from_directory(app.config['UPLOAD_FOLDER'], filename))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0'
        return response

def serve_mp4_with_range(file_path):
    print("Serving", file_path)
    """Serve MP4 video with support for range requests (chunking)."""
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range", None)

    if not range_header:
        # No range request → Send full file
        with open(file_path, "rb") as f:
            data = f.read()
        return Response(data, status=200, content_type="video/mp4")

    # **Handle range request for seeking**
    try:
        byte_range = range_header.split("=")[1]
        start, end = byte_range.split("-")
        start = int(start)
        end = int(end) if end else start + (1024 * 1024)  # Default to 1MB chunks
        end = min(end, file_size - 1)  # Avoid reading beyond file size
    except ValueError:
        return abort(400, "Invalid range request")

    chunk_size = end - start + 1
    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    response = Response(data, status=206, content_type="video/mp4")
    response.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Length"] = str(chunk_size)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0'
    return response

"""
@app.route('/get_conversation_ids')
def get_conversation_ids():
    user_id = request.args.get('user_id')  # Get the user_id from query parameters

    if user_id in conversations_db:
        conversation_ids = conversations_db[user_id]  # Retrieve conversation IDs
        return jsonify({"conversationIds": conversation_ids})  # Return as JSON
    else:
        return jsonify({"error": "User not found."}), 404  # Handle user not found case
"""
"""
def emit_back_to_client(user_id, response_data):
    sid = clients.get(user_id)
    if sid:
        socketio.emit('server_response', response_data, room=sid)
    else:
        print(f"Client {user_id} is not connected")
"""

@socketio.on('get_conversation_ids')
def handle_get_conversation_ids(data):
    user_id = data.get('user_id')
    
    global conversations
    #If not exists create duckdb database
    conversation_ids = list(conversations.get(user_id, {}).keys())
    #print("Conversation ids: ", conversation_ids, conversations)
    #We store conversation_ids of
    
    #Should retrieve 
    emit('conversation_ids', {'conversationIds': conversation_ids})

@socketio.on('get_past_messages')
def handle_get_past_messages(data):
    user_id = data['user_id']
    
    conversation_id = data['conversation_id']
    global conversations
    # Fetch past messages for the selected conversation
    #If not exist 
    past_messages = conversations.get(user_id, {}).get(conversation_id, [])
    print("Past messages: ", user_id, conversation_id, past_messages, conversations)
    # Send the past messages back to the client
    #We gotta also load the conversation in the past_conversations also save it everytime a new message is recieved
    #To do that we have to modify recieved message to save
    emit('past_messages', past_messages)

# GitHub API Configuration
GITHUB_TOKEN = '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'
GITHUB_REPO = 'sprites20/ngrok-links'
GITHUB_BRANCH = 'main'
NGROK_LINK_FILE_PATH = 'ngrok-link.json'

# Function to check if the file exists in the repository
def file_exists():
    file_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{NGROK_LINK_FILE_PATH}?ref={GITHUB_BRANCH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(file_url, headers=headers)
    return response.status_code == 200

# Function to create a new file in GitHub
def create_file(public_url):
    file_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{NGROK_LINK_FILE_PATH}?ref={GITHUB_BRANCH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Prepare the content to be written to the file
    # Construct the JSON object with the ngrok URL
    new_content = {
        "ngrok_url": public_url  # Set the URL in the JSON structure
    }

    # Encode the JSON object as Base64
    content_base64 = base64.b64encode(json.dumps(new_content).encode('utf-8')).decode('utf-8')  # Encode in Base64

    # Prepare the payload to create the file
    data = {
        "message": "Create ngrok-link.json",
        "content": content_base64,
        "branch": GITHUB_BRANCH
    }

    # Create the file on GitHub
    create_response = requests.put(file_url, headers=headers, json=data)
    if create_response.status_code == 201:
        print(f"Successfully created ngrok-link.json in GitHub with URL: {public_url}")
    else:
        print("Error creating file in GitHub:", create_response.text)

# Function to update the ngrok URL in GitHub
def update_github_ngrok_link(public_url):
    if file_exists():
        print("File exists. Proceeding to update.")
        file_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{NGROK_LINK_FILE_PATH}?ref={GITHUB_BRANCH}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Fetch the current file SHA
        response = requests.get(file_url, headers=headers)
        
        if response.status_code == 200:
            file_info = response.json()
            file_sha = file_info['sha']

            # Prepare the new content
            new_content = {
                "ngrok_url": public_url  # Set the URL in the JSON structure
            }
            
            # Encode the JSON object as Base64
            content_base64 = base64.b64encode(json.dumps(new_content).encode('utf-8')).decode('utf-8')  # Encode in Base64

            # Prepare the payload to update the file
            data = {
                "message": "Update ngrok URL",
                "content": content_base64,
                "sha": file_sha,
                "branch": GITHUB_BRANCH
            }

            # Update the file on GitHub
            update_response = requests.put(file_url, headers=headers, json=data)
            if update_response.status_code == 200:
                print(f"Successfully updated ngrok URL in GitHub to {public_url}")
            else:
                print("Error updating file in GitHub:", update_response.text)
        else:
            print("Error fetching file from GitHub:", response.text)
    else:
        print("File does not exist. Creating file.")
        create_file(public_url)


# Function to update the ngrok URL in GitHub
def update_github_ngrok_link(public_url):
    if file_exists():
        print("File exists. Proceeding to update.")
        file_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{NGROK_LINK_FILE_PATH}?ref={GITHUB_BRANCH}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Fetch the current file SHA
        response = requests.get(file_url, headers=headers)
        
        if response.status_code == 200:
            file_info = response.json()
            file_sha = file_info['sha']

            # Prepare the new content
            new_content = {
                "ngrok_url": public_url  # Set the URL in the JSON structure
            }
            
            # Encode the JSON object as Base64
            content_base64 = base64.b64encode(json.dumps(new_content).encode('utf-8')).decode('utf-8')  # Encode in Base64

            # Prepare the payload to update the file
            data = {
                "message": "Update ngrok URL",
                "content": content_base64,
                "sha": file_sha,
                "branch": GITHUB_BRANCH
            }

            # Update the file on GitHub
            update_response = requests.put(file_url, headers=headers, json=data)
            if update_response.status_code == 200:
                print(f"Successfully updated ngrok URL in GitHub to {public_url}")
            else:
                print("Error updating file in GitHub:", update_response.text)
        else:
            print("Error fetching file from GitHub:", response.text)
    else:
        print("File does not exist. Creating file.")
        create_file(public_url)


# Function to reconnect ngrok every 5 minutes and update GitHub
def update_ngrok_url_periodically():
    ngrok.set_auth_token("2hZ1a3ktCeJBhkG9DsThddItHbW_4rd6NSVgphvNE5Efti4A9")

    while True:
        # Open a new ngrok tunnel
        global public_url
        public_url = ngrok.connect(5000, bind_tls=True)
        print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")

        # Update the ngrok link in GitHub
        update_github_ngrok_link(str(public_url))

        # Wait for 5 minutes (300 seconds)
        time.sleep(60*60*5)

        # Disconnect the current ngrok tunnel
        ngrok.disconnect(public_url)

# Function to run the Flask-SocketIO server
def run_server():
    ngrok.set_auth_token("2hZ1a3ktCeJBhkG9DsThddItHbW_4rd6NSVgphvNE5Efti4A9")
    #public_url = ngrok.connect(5000)
    #print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
    
    # Start the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, use_reloader=False, log_output=True)



# Start the ngrok URL update thread
ngrok_update_thread = threading.Thread(target=update_ngrok_url_periodically)
ngrok_update_thread.daemon = True
ngrok_update_thread.start()

# Start the server in a separate thread
server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()

# Main program continues here
print("SocketIO server is running in the background and ngrok URL is being updated every 5 minutes.")

node_init = {
    "ignition" : {
            "function_name": "ignition",
            "import_string" : None,
            "function_string" : """
def ignition(node):
    print("Ignition")
    await asyncio.sleep(.25)
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
			"user_id" : "string",
            },
            "outputs" : {
			"user_id" : "string",
            }
        },
    "process_connection_queue" : {
        "function_name": "process_connection_queue",
        "import_string" : None,
        "function_string" : """
def process_connection_queue(node):
    if not connection_queue.empty():
        user_id, sid = connection_queue.get()  # Get the next connection request
        if user_id:
            clients[user_id] = sid  # Register the client
            print(f"Client {user_id} connected with session ID {sid}")
            node.args["user_id"] = user_id
            return {"user_id" : user_id}
        else:
            return None
    else:
        return None
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
        "user_id" : "string",
        },
        "outputs" : {
        "user_id" : "string",
        }
    },
    "process_queue" : {
        "function_name": "process_queue",
        "import_string" : None,
        "function_string" : """
def process_queue(node):
    if not user_queues.empty():
        message_data = user_queues.get()  # Get the oldest message in the queue
        if message_data:
            print("Message data: ", message_data)
            return {
                "user_id" : message_data["sender"], 
                "message" : message_data["text"],
                "image" : message_data["image"],
                "conversation_id" : message_data["conversation_id"]
            }
        else:
            return None
    else:
        return {
                "user_id" : None, 
                "message" : None,
                "image" : None,
                "conversation_id" : None,
            }
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
        "user_id" : "string",
        },
        "outputs" : {
            "user_id" : "string",
            "message" : "string",
            "image" : "image",
            "conversation_id" : "string"
        }
    },
    "emit_back_to_client" : {
        "function_name": "emit_back_to_client",
        "import_string" : None,
        "function_string" : """
def emit_back_to_client(node, user_id=None, message=None, image=None, conversation_id=None):
    print(user_id, message, image)
    print("Emitting to: ", user_id)
    sid = clients.get(user_id)
    image_url = "nil"
    response_data = {
        'text': message,
        'image': image_url,
        'sender': "timestamp_Bot",
        'conversation_id': conversation_id,
    }
    try:
        global conversations
        conversations[user_id][conversation_id].append(response_data)  # Store the message
    except:
        pass
    if sid:
        socketio.emit('server_response', response_data, room=sid)
    else:
        print(f"Client {user_id} is not connected")
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "user_id" : "string",
            "message" : "string",
            "image" : "string",
            "conversation_id" : "string"
        },
        "outputs" : {
            "user_id" : "string",
        }
    },
    "stt" : {
        "function_name": "stt",
        "import_string" : None,
        "function_string" : """
import os
import sys
import wave
import json
import numpy as np

import argparse
import queue
import sys
import sounddevice as sd
from pydub.silence import split_on_silence
from pydub import AudioSegment

from vosk import Model, KaldiRecognizer, SpkModel


import io
from faster_whisper import WhisperModel

import concurrent.futures

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    '''This is called (from a separate thread) for each audio block.'''
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata)) 

SPK_MODEL_PATH = 'vosk-model-spk-0.4'

if not os.path.exists(SPK_MODEL_PATH):
    print('Please download the speaker model from '
        'https://alphacephei.com/vosk/models and unpack as {SPK_MODEL_PATH} '
        'in the current folder.')
    sys.exit(1)
q = queue.Queue()
# Large vocabulary free form recognition
model = Model('vosk-model-small-en-us-0.15')
spk_model = SpkModel(SPK_MODEL_PATH)

model_size = 'small'
# Run on GPU with FP16
model_whisper = WhisperModel(model_size, device='cpu', compute_type='int8')
def wav_to_text(audio_path):
    timeout = 7
    def transcribe():
        segments, info = model_whisper.transcribe(audio_path, beam_size=5, word_timestamps=True)

        print('Detected language %s with probability %f', (info.language, info.language_probability))
        
        if info.language == 'en':
            transcribed_text = ''
            for segment in segments:
                for word in segment.words:
                    transcribed_text += word.word + ' '
                transcribed_text += ' '
                print(segment.words)
            print(transcribed_text)
            return transcribed_text

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(transcribe)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print('The transcription process timed out.')
            return None

            

# We compare speakers with cosine distance.
# We can keep one or several fingerprints for the speaker in a database
# to distingusih among users.
spk_sig = [-0.645543, 1.267236, 1.739462, -0.717491, -0.157087, 0.147635, -1.308505, -0.446466, 0.116764, -0.115046, 0.376392, 0.62511, 0.554749, 0.871882, 1.705446, 1.346732, -0.237086, 0.554086, 0.171249, 0.035732, 0.079214, -0.577399, 1.605019, -0.872605, -0.80465, -0.402827, -0.621014, -0.13613, 1.766777, 1.253641, -1.048572, -1.723634, -0.525028, -0.512419, 0.979154, -0.29935, -1.11108, 1.460288, -0.492389, -0.165662, -0.274988, 0.458642, 1.453099, 1.092062, -0.856726, 0.724769, 0.423962, -0.774903, -0.434743, -0.083244, 0.685712, -0.579763, -0.160493, 0.699621, -0.95782, -1.056444, -0.218858, 0.508616, -0.441598, 0.140081, 0.870923, -1.356405, -0.179892, -0.495612, -0.165929, 0.162548, -0.490384, 0.044856, -0.585081, 2.214094, 0.511557, -2.132176, -0.329827, 1.419002, -1.156591, -0.265651, -1.553596, -0.50643, 0.627002, -1.194909, -0.253832, 0.115579, 0.164481, -0.543525, -0.657609, 0.529603, 0.917261, 1.276905, 2.072457, 0.501246, -0.229274, 0.554694, -1.703213, -0.693821, 0.768317, -0.404479, 2.06889, -1.26462, -0.019318, 0.715243, 1.138082, -1.728924, -0.714421, -1.267921, 1.681902, -1.716266, -0.074632, -2.936986, -2.350122, 0.001327, -0.382891, -0.688902, 1.322296, -0.987495, 1.975746, -0.44887, 0.185008, 0.067595, 0.665363, 0.246385, 0.719629, 0.506032, -0.988654, 0.606328, -1.949532, 1.727559, -1.032074, -0.772542]

def cosine_similarity_average(speaker_embeddings, target_speaker):
    lowest_similarity = {}

    for speaker, embeddings in speaker_embeddings.items():
        # Get the lowest similarity among the two embeddings for each speaker
        similarities = [cosine_dist(target_speaker, embedding) for embedding in embeddings]
        lowest_similarity[speaker] = min(similarities)

    return lowest_similarity

def recognize_speaker(target_speaker):
    speakers = {
        'speaker1': [[-0.645543, 1.267236, 1.739462, -0.717491, -0.157087, 0.147635, -1.308505, -0.446466, 0.116764, -0.115046, 0.376392, 0.62511, 0.554749, 0.871882, 1.705446, 1.346732, -0.237086, 0.554086, 0.171249, 0.035732, 0.079214, -0.577399, 1.605019, -0.872605, -0.80465, -0.402827, -0.621014, -0.13613, 1.766777, 1.253641, -1.048572, -1.723634, -0.525028, -0.512419, 0.979154, -0.29935, -1.11108, 1.460288, -0.492389, -0.165662, -0.274988, 0.458642, 1.453099, 1.092062, -0.856726, 0.724769, 0.423962, -0.774903, -0.434743, -0.083244, 0.685712, -0.579763, -0.160493, 0.699621, -0.95782, -1.056444, -0.218858, 0.508616, -0.441598, 0.140081, 0.870923, -1.356405, -0.179892, -0.495612, -0.165929, 0.162548, -0.490384, 0.044856, -0.585081, 2.214094, 0.511557, -2.132176, -0.329827, 1.419002, -1.156591, -0.265651, -1.553596, -0.50643, 0.627002, -1.194909, -0.253832, 0.115579, 0.164481, -0.543525, -0.657609, 0.529603, 0.917261, 1.276905, 2.072457, 0.501246, -0.229274, 0.554694, -1.703213, -0.693821, 0.768317, -0.404479, 2.06889, -1.26462, -0.019318, 0.715243, 1.138082, -1.728924, -0.714421, -1.267921, 1.681902, -1.716266, -0.074632, -2.936986, -2.350122, 0.001327, -0.382891, -0.688902, 1.322296, -0.987495, 1.975746, -0.44887, 0.185008, 0.067595, 0.665363, 0.246385, 0.719629, 0.506032, -0.988654, 0.606328, -1.949532, 1.727559, -1.032074, -0.772542],
                [-0.683516, 0.722179, 1.651159, -0.311776, -0.35272, -0.542711, -0.169784, 0.146419, 0.639174, 0.260786, 0.512685, -0.567375, 0.510885, 1.081993, 0.730045, 1.644301, -0.388575, 0.594761, 0.580934, 1.701163, 0.542753, -0.030902, 0.940672, -0.681181, -0.961269, -0.953732, 0.342842, 0.212761, 1.010038, 0.789226, -0.440633, -1.639356, 0.098124, -0.453873, -0.1269, -0.831008, -1.336311, 1.838328, -1.500506, 0.398561, -0.139225, 0.602066, 1.217693, -0.28669, -1.240536, 0.828214, -0.385781, -1.585939, -0.253948, 0.6254, -1.144157, -1.09649, -1.247936, -0.164992, -1.131125, -0.827816, 1.595752, 1.22196, -0.260766, -0.053225, 0.372862, -0.496685, 0.559101, 0.313831, 0.906749, -0.911119, -0.718342, 0.731359, -0.060828, 0.889468, 0.870002, -1.046849, 0.358473, 1.403957, -0.55995, 0.544278, 0.252579, 0.176449, -0.973618, -1.316356, -1.39273, -0.397281, -1.244906, -2.552846, -0.056479, 0.00252, -0.071661, 0.549343, -0.563582, 0.298601, -1.599536, 0.060805, -1.131684, -0.236406, 0.10192, -0.05143, 2.822287, 0.298605, 0.027687, 1.805171, 0.535367, -0.750344, 0.195215, -2.74342, -0.240448, -1.853602, 0.667115, -1.152912, -1.458451, -0.463823, -1.081316, 1.07476, 1.69582, 0.083853, 0.208222, -0.203687, -0.761975, 2.021879, 2.07578, 0.214109, 1.010975, -0.535104, -1.102454, 1.422523, -1.389488, 2.282245, 0.526214, -0.289677],
                [-0.645543, 1.267236, 1.739462, -0.717491, -0.157087, 0.147635, -1.308505, -0.446466, 0.116764, -0.115046, 0.376392, 0.62511, 0.554749, 0.871882, 1.705446, 1.346732, -0.237086, 0.554086, 0.171249, 0.035732, 0.079214, -0.577399, 1.605019, -0.872605, -0.80465, -0.402827, -0.621014, -0.13613, 1.766777, 1.253641, -1.048572, -1.723634, -0.525028, -0.512419, 0.979154, -0.29935, -1.11108, 1.460288, -0.492389, -0.165662, -0.274988, 0.458642, 1.453099, 1.092062, -0.856726, 0.724769, 0.423962, -0.774903, -0.434743, -0.083244, 0.685712, -0.579763, -0.160493, 0.699621, -0.95782, -1.056444, -0.218858, 0.508616, -0.441598, 0.140081, 0.870923, -1.356405, -0.179892, -0.495612, -0.165929, 0.162548, -0.490384, 0.044856, -0.585081, 2.214094, 0.511557, -2.132176, -0.329827, 1.419002, -1.156591, -0.265651, -1.553596, -0.50643, 0.627002, -1.194909, -0.253832, 0.115579, 0.164481, -0.543525, -0.657609, 0.529603, 0.917261, 1.276905, 2.072457, 0.501246, -0.229274, 0.554694, -1.703213, -0.693821, 0.768317, -0.404479, 2.06889, -1.26462, -0.019318, 0.715243, 1.138082, -1.728924, -0.714421, -1.267921, 1.681902, -1.716266, -0.074632, -2.936986, -2.350122, 0.001327, -0.382891, -0.688902, 1.322296, -0.987495, 1.975746, -0.44887, 0.185008, 0.067595, 0.665363, 0.246385, 0.719629, 0.506032, -0.988654, 0.606328, -1.949532, 1.727559, -1.032074, -0.772542],
            
        ],
    }
    #Iterate in speakers and get average cosine similarity and return average cosine similarity for each and the lowest
    lowest_similarity = cosine_similarity_average(speakers, target_speaker)

    # Print the results
    print('Lowest Similarities:')
    dspeaker, thesim = None, None
    for speaker, lowest_sim in lowest_similarity.items():
        print(f'{speaker}: {lowest_sim}')
        dspeaker, thesim = speaker, lowest_sim
        break
    
    #print(cosine_similarity_average(speakers, target_speaker))
    if thesim > 0.55:
        dspeaker = 'Unknown User'
    return dspeaker
def cosine_dist(x, y):
    nx = np.array(x)
    ny = np.array(y)
    return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)
def save_audio(filename, audio_data, samplerate):
    audio_segment = AudioSegment(
        data=audio_data, sample_width=2, frame_rate=samplerate, channels=1
    )
    audio_segment.export(filename, format='wav')
# Function to modify already saved audio file
def modify_saved_audio(input_filename, output_filename, samplerate):
    audio_segment = AudioSegment.from_wav(input_filename)

    # Split the audio based on silence
    # Adjust silence detection parameters as needed
    segments = split_on_silence(audio_segment, silence_thresh=-40, keep_silence=100)

    # Concatenate non-silent segments
    concatenated_audio = AudioSegment.silent()
    for i, segment in enumerate(segments):
        concatenated_audio += segment

    # Save the concatenated audio
    concatenated_audio.export(output_filename, format='wav')

#Create toggle button for STT module

#Initialize these in the node
#Create node that will initialize these and pass through the STT module
device_info = sd.query_devices(None, 'input')
samplerate = int(device_info['default_samplerate'])

rec = KaldiRecognizer(model, samplerate)
rec.SetSpkModel(spk_model)

stream = sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None,
                           dtype='int16', channels=1, callback=callback)
stream_started = False
#stream.start()

recording_started = False
audio_data = b''

wait_time = 0
spoken = False



def stt(node):
    def update_text_input(dt, transcribed_text):
        app.root.get_screen("chatbox").ids.text_input.text += transcribed_text

    await asyncio.sleep(0.1)
    global stream_started
    global wait_time
    global spoken
    
    if not stream_started:
        stream_started = True
        stream.start()
        
    wait_time += 0.1
    #Connect this and return 
    data = q.get()
    transcribed_text = ''
    global recording_started
    global audio_data
    
    app = MDApp.get_running_app()
    if rec.PartialResult() != '':
        if not recording_started:
            print('Recording started.')
            recording_started = True

    if recording_started:
        audio_data += data

    if rec.AcceptWaveform(data):
        res = json.loads(rec.Result())
        print('Text:', res['text'])
        print('Recording stopped.')
        result = res.get('text')
        print(result)
        # self.change_text_input(result)
        save_audio('output_audio.wav', audio_data, samplerate)
        
        # After recording is done and you have an output audio file
        input_audio_filename = 'output_audio.wav'
        
        
        if result:
            transcribed_text = wav_to_text('output_audio.wav')
            # Schedule the update after 0 seconds (or you can specify a delay)
            Clock.schedule_once(lambda dt: update_text_input(dt, transcribed_text))
            wait_time = 0
            spoken = True
        #Connect this and return text
        
        #Connect and 
        recording_started = False
        audio_data = b''
        if 'spk' in res:
            recognize_speaker(res['spk'])
            print('X-vector:', res['spk'])
    print(transcribed_text)
    return {'transcribed_text' : transcribed_text}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "transcribed_text" : "string",
        }
    },
    "text_to_wav_instance" : {
        "function_name": "text_to_wav_instance",
        "import_string" : None,
        "function_string" : """
'''
def text_to_wav_instance(node, text):
    return None
'''

import time
import wave

tts = None
try:
    from TTS.api import TTS
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
engine = pyttsx3.init()
# Set the rate and volume (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
def text_to_wav_instance(node, text):
    filename = "output.wav"
    global engine
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
        engine.save_to_file(text, filename)
        engine.runAndWait()
        
        sound = SoundLoader.load(filename)
        node.args["sound"] = sound
        return {"speech_wav" : sound}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "text" : "string"
        },
        "outputs" : {
			"user_id" : "string",
            "speech_wav" : "sound",
            "duration" : "num"
        }
    },
    "play_audio_tts" : {
        "function_name": "play_audio_tts",
        "import_string" : None,
        "function_string" : """
def play_audio_tts(node, sound=None, duration=None):
    '''
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
    '''
    if sound:
        sound.play()
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "sound" : "sound",
            "duration" : "num"
        },
        "outputs" : {
			"user_id" : "string",
        }
    },
    "stop_audio_tts" : {
        "function_name": "stop_audio_tts",
        "import_string" : None,
        "function_string" : """
def stop_audio_tts(node, sound=None):
    try:
        if sound.state == 'play':
            sound.stop()
    except:
        pass
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "sound" : "sound"
        },
        "outputs" : {
			"user_id" : "string",
        }
    },
    "reset_input_box" : {
        "function_name": "reset_input_box",
        "import_string" : None,
        "function_string" : """

def reset_input_box(node):
    app = MDApp.get_running_app()
    def update_text_input(dt):
        app.root.get_screen("chatbox").ids.text_input.text = ''
    Clock.schedule_once(lambda dt: update_text_input(dt))
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
        },
        "outputs" : {
			"user_id" : "string",
        }
    },
    "trigger_after_stt" : {
        "function_name": "trigger_after_stt",
        "import_string" : None,
        "function_string" : """
def trigger_after_stt(node, transcribed_text=None):
    global wait_time
    global spoken
    if spoken:
        print("Spoken checkpoint")
        if wait_time >= 3:
            print("Wait time is: ", wait_time)
            wait_time = 0
            spoken = False
    else:
        print("Stopped: trigger_after_stt")
        node.stop = True
        return None
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "transcribed_text" : "string"
        },
        "outputs" : {
			"user_id" : "string",
        }
    },
    "search_facebook" : {
        "function_name": "search_facebook",
        "import_string" : None,
        "function_string" : """

def search_facebook(node, user_input=None, instruct_type=None, user_id=None, conversation_id=None, image=None):
    if instruct_type == 3:
        #Generate facebook prompt
        
        # Building the command
        command = ['python', 'search_facebook.py'] + [user_input]
        
        # Using subprocess to run the command and capture the output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #print(result)
        result = str(result.stdout)
        #print(result)
        # Return the stdout and stderr
        return {'output': result}
    else:
        node.stop = True
        return None
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "user_input" : "string",
            "instruct_type" : "num",
        },
        "outputs" : {
			"user_id" : "string",
            "output" : "output",
        }
    },
    "file_chooser" : {
        "function_name": "file_chooser",
        "import_string" : None,
        "function_string" : """
def file_chooser(node):
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
			"user_id" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "dir" : "string"
        }
    },
    "image_chooser" : {
        "function_name": "image_chooser",
        "import_string" : None,
        "function_string" : """
def image_chooser(node):
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
			"user_id" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "user_image" : "string"
        }
    },
    "display_output" : {
        "function_name": "display_output",
        "import_string" : None,
        "function_string" : """
def display_output(node, user_input=None, output=None, instruct_type=None, generated_image_path=None, user_image=None):
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
        print("User Image: ", user_image)
        if user_image:
            print(user_image)
            grid_layout.add_widget(CustomImageComponent(img_source=user_image))
        grid_layout.add_widget(bot_custom_component)
        
        if instruct_type == 2:
            #image_components.append(CustomImageComponent(img_source=user_image))
            grid_layout.add_widget(CustomImageComponent(img_source=generated_image_path))
        
    # Schedule the update_ui function to run on the main thread
    Clock.schedule_once(update_ui)
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "user_input" : "string",
            "output" : "string",
            "instruct_type" : "num",
            "generated_image_path" : "string",
            "user_image" : "string",
        },
        "outputs" : {
			"user_id" : "string",
        }
    },

    "select_model" : {
            "function_name": "select_model",
            "import_string" : None,
            "function_string" : """
def select_model(node):
    print("select_model")
    await asyncio.sleep(.25)
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
			"user_id" : "string",
            },
            "outputs" : {
			"user_id" : "string",
                "model" : "string",
            }
        },
    "user_input" : {
            "function_name": "user_input",
            "import_string" : None,
            "function_string" : '''
def user_input(node):
    try:
        global spoken
        print("Spoken: ", spoken)
        if not spoken:
            spoken = False
            app = MDApp.get_running_app()
            user_input = app.root.get_screen("chatbox").ids.text_input.text
            print("Printing user_input: ", user_input)
            return {"user_input" : user_input}
        else:
            node.stop = True
    except:
        app = MDApp.get_running_app()
        user_input = app.root.get_screen("chatbox").ids.text_input.text
        print("Printing user_input: ", user_input)
        return {"user_input" : user_input}
    
            ''',
            "description" : None,
            "documentation" : None,
            "inputs" : {
			"user_id" : "string",
            },
            "outputs" : {
			"user_id" : "string",
                "user_input" : "string",
            }
        },
    "pass_node" : {
            "function_name": "pass_node",
            "import_string" : None,
            "function_string" : """
def pass_node(node):
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
			"user_id" : "string",
            },
            "outputs" : {
			"user_id" : "string",
            }
        },
    "context" : {
            "function_name": "context",
            "import_string" : None,
            "function_string" : """
def context(node):
    print("context")
    await asyncio.sleep(.25)
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
			"user_id" : "string",
            },
            "outputs" : {
			"user_id" : "string",
                "context" : "string",
            }
        },
    "reset_outputs" : {
            "function_name": "reset_outputs",
            "import_string" : None,
            "function_string" : """
def reset_outputs(node):
    return None
            """,
            "description" : None,
            "documentation" : None,
            "inputs" : {
			"user_id" : "string",
            },
            "outputs" : {
			"user_id" : "string",
            }
        },
    "is_greater_than" : {
        "function_name": "is_greater_than",
        "import_string" : None,
        "function_string" : """
def is_greater_than(node, A=None, B=None):
    return {"is_greater_than" : A > B}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "A" : "bool",
            "B" : "bool"
        },
        "outputs" : {
			"user_id" : "string",
            "is_greater_than" : "bool"
        }
    },
    "is_less_than" : {
        "function_name": "is_less_than",
        "import_string" : None,
        "function_string" : """
def is_less_than(node, A=None, B=None):
    return {"is_less_than" : A < B}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "A" : "bool",
            "B" : "bool"
        },
        "outputs" : {
			"user_id" : "string",
            "is_less_than" : "bool"
        }
    },
    "is_equal" : {
        "function_name": "is_equal",
        "import_string" : None,
        "function_string" : """
def is_equal(node, A=None, B=None):
    return {"is_equal" : A == B}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "A" : "bool",
            "B" : "bool"
        },
        "outputs" : {
			"user_id" : "string",
            "is_equal" : "bool"
        }
    },
    "get_instruct_type_node" : {
        "function_name": "get_instruct_type_node",
        "import_string" : None,
        "function_string" : '''
def get_instruct_type_node(node, user_id=None, user_input=None, context=None, image=None, conversation_id=None):
    # Input text containing the Python code block
    #print("Running get instruct type")
    #asyncio.sleep(0.1)
    
    if user_id == None or user_input == None:
        node.stop["user"] = True
        #print("None, skipping")
        return None
        
    generate_code = (
        f"User Input: {user_input}\\n"
        "Instruct Types:\\n"
        "0: Error, if empty user input or just blank spaces, return this\\n"
        "1: Normal, normal conversation\\n"
        "2: Generate Image, if user wants to generate an image\\n"
        "3: Search Facebook, if user wants to search Facebook.\\n"
        "4: Search Google, If user wants to do Web Search or if you dont know the answer or wants updated answer.\\n"
        "5: Search Google with Images, If user wants to search images of an object\\n"
        "First justify why, then output only the number of the instruct type, with format: \\n"
        "Format: instruct type:<number>"
    )
    message_array = []
    message_array.append({"role": "system", "content": "Your role is to decide what the instruct type to use based on the user intent. "})
    message_array.append({"role": "user", "content": generate_code})
    if context:
        message_array.append({"role": "context", "content": context})
    global TOGETHER_API_KEY
    
    chat_completion = client_for_instruct.chat.completions.create(
      messages=message_array,
      model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    )
    
    response = chat_completion.choices[0].message.content
    print(response)
    # Use regular expression to find the instruct_type number
    pattern = re.compile(r"instruct type\\s*:\\s*(\\d+)")
    # Convert the string to all lowercase
    response = response.lower()
    match = pattern.search(response)
    instruct_type = None
    if match:
        instruct_type = int(match.group(1))
        print(f"instruct_type number: {instruct_type}")
    else:
        print("instruct_type not found in the data")
    #instruct_type = int(response)
    print("Bot: ", response, user_id, user_input)
    time.sleep(1.5)
    
    return {"user_id" : user_id, "instruct_type" : instruct_type, "message" : user_input, "image" : image, "conversation_id" : conversation_id}
        ''',
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "user_input" : "string",
            "context" : "string",
            "image" : "string",
            "conversation_id" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "message" : "string",
            "instruct_type" : "num",
            "context" : "string",
            "image" : "string",
            "conversation_id" : "string",
        }
    },
    "generate_image_prompt" : {
        "function_name": "generate_image_prompt",
        "import_string" : None,
        "function_string" : """
def generate_image_prompt(node, model=None, user_input=None, context=None, instruct_type=None):
    app = MDApp.get_running_app()
    print("Prompt")
    user_text = user_input
    if context:
        context = "OCR output:\\n" + context
        print("context: ", context)
    generated_image_path = ""
    # Continue the conversation            
    response = app.continue_conversation(user_text=user_text, context=context, user_id=user_id)
    print("output: ", response)
    return {"output" : response, "generated_image_path" : generated_image_path}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "model" : "string",
            "user_input" : "string", 
            "instruct_type" : "num",
            "context" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "output" : "string",
            "instruct_type" : "num",
        }
    },
    "is_normal_prompt" : {
        "function_name": "is_normal_prompt",
        "import_string" : None,
        "function_string" : """
def is_normal_prompt(node, user_id=None, message=None, instruct_type=None, image=None, conversation_id=None, context=None):
    print("user_id: ", user_id)
    print("Printed instruct: ", instruct_type)
    #if not instruct_type == 1:
        #node.stop = True
        #pass
    print("Ran is normal, message: ", message)
    return {"user_id" : user_id, "message" : message, "context" : context, "image" : image, "conversation_id" : conversation_id}
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "message" : "string",
            "instruct_type" : "num",
            "image" : "string",
            "conversation_id" : "string",
            "context" : "string"
            
        },
        "outputs" : {
			"user_id" : "string",
            "message" : "string",
            "image" : "string",
            "conversation_id" : "string",
            "context" : "string"
            
        }
    },
    "prompt" : {
        "function_name": "prompt",
        "import_string" : None,
        "function_string" : """
def prompt(node, user_id=None, model=None, user_input=None, context=None, instruct_type=None, image=None, conversation_id=None):
    app = MDApp.get_running_app()
    print("Ran Prompt")
    
    user_text = user_input
    # Continue the conversation
    if context:
        context = "Context: " + context
    if user_input != None:
        response = app.continue_conversation(user_text=user_text, context=context, user_id=user_id, image=image, conversation_id=conversation_id)
        print("output: ", response)
        time.sleep(1)
        print("Returning", model, user_id, user_input, context, response)
        return {"user_id" : user_id, "output" : response, "image" : image, "conversation_id" : conversation_id}
    else:
        node.stop[user_id] = True
        return None
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "model" : "string",
            "user_input" : "string", 
            "instruct_type" : "num",
            "context" : "string",
            "image" : "string",
            "conversation_id" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "output" : "string",
            "image" : "string",
            "conversation_id" : "string",
        }
    },
    "image_to_text" : {
        "function_name": "image_to_text",
        "import_string" : None,
        "function_string" : """
def image_to_text(node, user_id=None, user_input=None, context=None, image=None):
    # Input text containing the Python code block
    #print("Running get instruct type")
    asyncio.sleep(0.1)
    if user_id == None or user_input == None:
        node.stop["user"] = True
        return None
    if user_image:
        # Load the image using PIL
        print(user_image)
        # Convert the image to a format OpenCV can work with
        image = cv2.imread(user_image)
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use pytesseract to get detailed OCR results
        detailed_data = pytesseract.image_to_data(image_cv, output_type=pytesseract.Output.DICT)
        
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
            if conf > 20:
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
			"user_id" : "string",
            "user_image" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "output_text" : "string",
        }
    },
    "translate_language" : {
        "function_name": "translate_language",
        "import_string" : None,
        "function_string" : """
def translate_language(node, input_text=None, input_language=None, output_language="English"):
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
			"user_id" : "string",
            "input_text" : "string",
            "input_language" : "string",
            "output_language" : "string"
         },
        "outputs" : {
			"user_id" : "string",
            "output_text" : "string",
        }
    },
    "detect_language" : {
        "function_name": "detect_language",
        "import_string" : None,
        "function_string" : """
def detect_language(node, input_text=None):
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
			"user_id" : "string",
            "input_text" : "string",
        },
        "outputs" : {
			"user_id" : "string",
            "language" : "string",
        }
    },
    "delay" : {
        "function_name": "delay",
        "import_string" : None,
        "function_string" : """
def delay(node, delay_seconds=None):
    if node.trigger_in.startswith("time_delta_seconds"):
        
        asyncio.sleep(delay_seconds)
    elif delay_seconds:
        asyncio.sleep(delay_seconds)
    return None
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
			"user_id" : "string",
            "delay_seconds": "string",
        },
        "outputs" : {
			"user_id" : "string",
        }
    },
    "time_delta_seconds_from_now" : {
        "function_name": "time_delta_seconds",
        "import_string" : None,
        "function_string" : """
def time_delta_seconds(node, given_date_time_str):
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
			"user_id" : "string",
            "given_date_time_str": "string",
        },
        "outputs" : {
			"user_id" : "string",
            "seconds" : "num",
        }
    },
    "decide_output_language" : {
        "function_name": "decide_output_language",
        "import_string" : None,
        "function_string" : """
def decide_output_language(node, user_language=None, listener_language=None, user_prompt=None, user_info=None, listener_info=None):
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
			"user_id" : "string",
            "user_prompt": "string",
            "user_language": "string",
            "user_info": "string",
            "listener_language": "string",
            "listener_info": "string"
        },
        "outputs" : {
			"user_id" : "string",
            "language" : "string",
        }
    },   
}

#node_init = {}

def load_json_files_to_dict(directory):
    path = Path(directory)
    data_dict = {}
    
    # Iterate through all JSON files in the directory
    for json_file in path.rglob("*.json"):
        with json_file.open('r') as f:
            # Load JSON data
            data = json.load(f)
            # Store data in the dictionary with file name (without extension) as the key
            data_dict[json_file.stem] = data
    
    return data_dict

# Example usage
directory_path = "nodes"  # Directory containing the JSON files

# Load JSON files into a dictionary
#node_init = load_json_files_to_dict(directory_path)

# Example usage
output_directory_path = "nodes"

# Save each dictionary to separate JSON files
#save_dicts_as_json_files(node_init, output_directory_path)
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

<MapScreen>
    name: 'map_screen'
    
<NewNodeScreen>
    name: 'new_node_screen'

<SelectNodeScreen>
    name: 'select_node_screen'

<WidgetTreeScreen>
    name: 'widget_tree_screen'

<SelectAppScreen>
    name: 'select_app_screen'

<LoraTrainerScreen>
    name: 'lora_trainer_screen'

<KukaScreen>
    name: 'kuka_screen'
    
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
                        #on_release: app.button_pressed()  # Define the action to be taken when the button is released
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
    LoraTrainerScreen:
    ChatboxScreen:
    SelectNodeScreen:
    RenderScreen:
    NewNodeScreen:
    WidgetTreeScreen:
    SelectAppScreen:
    MapScreen:
    KukaScreen:
'''

import math
import os
import requests
import asyncio
from functools import partial
#from pyppeteer import launch
import base64
from geopy.geocoders import Nominatim

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

from pyproj import Proj, transform

# Initialize the Nominatim geolocator with a custom user agent
geolocator = Nominatim(user_agent="my_app")

# Initialize global offset by the center of the screen
global_offset = [0,0]
# Dictionary to store images
images_dict = {}

class CircleWidget(Widget):
    def __init__(self, latitude, longitude, radius_pixels, zoom, mode, **kwargs):
        super(CircleWidget, self).__init__(**kwargs)
        self.latitude = latitude
        self.longitude = longitude
        self.radius_pixels = radius_pixels
        self.zoom = zoom
        self.mode = mode
        self.draw_circle()
 
    def draw_circle(self):
        with self.canvas:
            if self.mode == 0:
                Color(1, 0, 0, 1)  # Red color with 100% opacity
            elif self.mode == 1:
                Color(0, 0, 1, 1)
                self.radius_pixels = 3
            elif self.mode == 2:
                Color(0, 1, 0, 1)
                self.radius_pixels = 4
            elif self.mode == 3:
                Color(1, 0, 0, 1)
                self.radius_pixels = 3
                self.latitude -= .00035
            # Convert latitude and longitude to tile pixel coordinates
            x_tile, y_tile, x_pixel, y_pixel = lat_lon_to_tile_pixel_with_pixel(self.latitude, self.longitude, self.zoom)
            # Draw the circle as an ellipse
            print(x_tile, y_tile, x_pixel, y_pixel)
            circle_x, circle_y = images_dict[(self.zoom, x_tile, y_tile)].global_x + x_pixel + Window.width / 2 - self.radius_pixels + global_offset[0], images_dict[(self.zoom, x_tile, y_tile)].global_y - y_pixel + Window.height / 2 - self.radius_pixels + global_offset[1]
            self.circle_ellipse = Ellipse(pos=(circle_x, circle_y),
                    size=(self.radius_pixels * 2, self.radius_pixels * 2))
            
            print("Drawn Circle at", circle_x, circle_y)
    def get_pos(self):
        return self.circle_ellipse.pos
    def update_position(self, dx, dy):
        #print("Updated")
        self.circle_ellipse.pos = (self.circle_ellipse.pos[0] + dx, self.circle_ellipse.pos[1] + dy)

class PathLine(Widget):
    def __init__(self, point1, point2, **kwargs):
        super(PathLine, self).__init__(**kwargs)
        self.point1 = point1
        self.point2 = point2
        self.line = None
        self.draw_line()

    def draw_line(self):
        with self.canvas:
            Color(1, 0, 0, 1)  # Red color with 100% opacity
            points = [self.point1[0] + 3, self.point1[1] + 3, self.point2[0] + 3, self.point2[1] + 3]
            self.line = Line(points=points, width=2)

    def update_position(self, dx, dy):
        if self.line:
            # Update each point of the line by dx and dy
            new_points = [
                self.line.points[0] + dx, self.line.points[1] + dy,  # Update first point
                self.line.points[2] + dx, self.line.points[3] + dy   # Update second point
            ]
            self.line.points = new_points

from kivy.uix.image import Image
     
class DraggableImage(Image):
    def __init__(self, offset_x=0, offset_y=0, zoom=None, x_tile=None, y_tile=None, **kwargs):
        # Ensure to pass all the keyword arguments to the Image base class
        super(DraggableImage, self).__init__(**kwargs)
        self.dragging = True
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.zoom = zoom
        self.x_tile = x_tile
        self.y_tile = y_tile
        self.update_position()
        self.dx = 0
        self.dy = 0
        self.init_x = self.x
        self.init_y = self.y
        self.global_x = self.x
        self.global_y = self.y
        
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Convert touch position to image coordinates
            
            image_x = touch.pos[0] - self.pos[0]
            image_y = touch.pos[1] - self.pos[1]
            #print(f"Clicked inside the image at: ({image_x}, {image_y})", self.x_tile, self.y_tile)
            #print(f"Clicked at: ({image_x}, {image_y})")
            latitude, longitude = tile_pixel_to_lat_lon(self.zoom, self.x_tile, self.y_tile, image_x, image_y)
            #print(f"Coords: {latitude}, {longitude}")
            self.dragging = True
            self.touch_offset = (self.x - touch.x, self.y - touch.y)
            self.dx = 0
            self.dy = 0
            touch.grab(self)  # Grab the touch event for this widget
            #print(self.x/256)
            return True
        return super(DraggableImage, self).on_touch_down(touch)
        
    def on_touch_move(self, touch):
        if self.dragging:
            # Calculate the movement distance
            self.dx = touch.dx
            self.dy = touch.dy
            #print(f"Moved: ({self.dx}, {self.dy})")
            
            # Convert touch position to image coordinates
            image_x = touch.pos[0] - self.pos[0]
            image_y = touch.pos[1] - self.pos[1]
            #latitude, longitude = tile_pixel_to_lat_lon(self.zoom, self.x_tile, self.y_tile, image_x, image_y)
            #print("Tile:", lat_lon_to_tile_pixel(latitude, longitude, self.zoom))
            #print(f"Latitude: {latitude}, Longitude: {longitude}")
            #print(f"Tile offset: x: {self.dx//256}, y: {self.dy//256}")
            
            # Move all tiles in the parent DraggableMapScreen
            """
            for tile in draggable_map.tiles:
                tile.x += dx
                tile.y += dy
                tile.dragging = False
            self.dragging = True
            """
            return True
        return super(DraggableImage, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.dragging and touch.grab_current == self:
            #self.dragging = False
            #print(f"Image Position: zoom = {self.zoom}, x={self.x}, y={self.y}, offset_x={self.offset_x}, offset_y={self.offset_y}")
            #print(self.zoom)
            #self.update_position()  # Update position after releasing
            pass
        return super(DraggableImage, self).on_touch_up(touch)

    def update_position(self):
        # This method updates the image position including the offset.
        self.pos = (self.x + self.offset_x, self.y + self.offset_y)
        print(f"Image Position: zoom = {self.zoom}, x={self.x}, y={self.y}, offset_x={self.offset_x}, offset_y={self.offset_y}")


class DraggableMapScreen(FloatLayout):
    def __init__(self, **kwargs):
        super(DraggableMapScreen, self).__init__(**kwargs)
        self.touch_x = 0
        self.touch_y = 0
        self.tiles = []
        self.circles = []  # Keep track of circle widgets
        self.path_lines = []
        
        self.reverse_geocoding = False
        
    def add_circle(self, latitude, longitude, zoom, mode):
        circle_widget = CircleWidget(latitude, longitude, 5, zoom, mode)
        self.add_widget(circle_widget, index=0)
        self.circles.append(circle_widget)  # Add to the list of circles
        print("Added Circle")
        return circle_widget.get_pos()
        
    def add_path_line(self, point1, point2):
        path_line_widget = PathLine(point1, point2)
        self.add_widget(path_line_widget)
        self.path_lines.append(path_line_widget)
    
    def update_path_line_positions(self, dx, dy):
        for line in self.path_lines:
            line.update_position(dx, dy)
    def update_circle_positions(self, dx, dy):
        for circle in self.circles:
            circle.update_position(dx, dy)
            
    # Function to perform reverse geocoding
    def reverse_geocode(self, latitude, longitude):
        location = geolocator.reverse((latitude, longitude))
        return location.address if location else None
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        return distance

    def query_osm_places_within_radius(self, latitude, longitude, radius, tags=None):
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
           [out:json];
           (
               node{";".join([f'["{k}"="{v}"]' for k, v in tags.items()])}(around:{radius},{latitude},{longitude});
               way{";".join([f'["{k}"="{v}"]' for k, v in tags.items()])}(around:{radius},{latitude},{longitude});
               relation{";".join([f'["{k}"="{v}"]' for k, v in tags.items()])}(around:{radius},{latitude},{longitude});
           );
           out body;
           >;
           out skel qt;
        """

        response = requests.get(overpass_url, params={"data": query})
        return response.json()
    def get_route_coords(self, origin, destination):
        # Define the bounding box or the center and distance to get the street network
        location_point = origin  # Example location point
        G = ox.graph_from_point(location_point, dist=1000, network_type='drive')

        
        # Find the nearest nodes to the origin and destination points
        origin_node = ox.distance.nearest_nodes(G, origin[1], origin[0])
        destination_node = ox.distance.nearest_nodes(G, destination[1], destination[0])

        # Find the shortest path between the nodes
        route = nx.shortest_path(G, origin_node, destination_node, weight='length')
        print("Found path")

        # Extract the coordinate sequence for the path
        route_nodes = [G.nodes[node] for node in route]
        route_coords = [(node['y'], node['x']) for node in route_nodes]

        # Print the coordinate sequence
        print("Route coordinates:")
        for coord in route_coords:
            print(coord)
        return route_coords, G, route
    def wgs84_to_web_mercator(self, lat, lon):
        # Ensure latitude is within the valid range
        lat = max(min(lat, 89.99999), -89.99999)
        
        # Convert latitude and longitude to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # Earth's radius in meters
        R = 6378137.0
        
        # Calculate the Web Mercator x and y coordinates
        x = lon_rad * R
        y = math.log(math.tan(math.pi / 4 + lat_rad / 2)) * R
        
        return x, y  # Return longitude (x) and latitude (y)
    
    def web_mercator_to_degrees(self, x, y):
        # Earth's radius in meters
        R = 6378137.0

        # Convert x and y from meters to radians
        lon = math.degrees(x / R)
        lat = math.degrees(math.atan(math.sinh(y / R))) - .001

        return lat, lon  # Return latitude and longitude in degrees
    def total_route_distance(self, route_coords):
        total_distance = 0.0
        for i in range(len(route_coords) - 1):
            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i + 1]
            segment_distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += segment_distance
            print(i, total_distance)
        return total_distance
    def on_touch_down(self, touch):
        global global_offset
        for child in self.children:
            if isinstance(child, DraggableImage):
                # Calculate the global position of the touch
                global_pos_x = touch.pos[0] - global_offset[0] - Window.width / 2
                global_pos_y = touch.pos[1] - global_offset[1] - Window.height / 2
                
                # Calculate the relative position of the touch inside the child
                touch_x_rel = global_pos_x - child.global_x
                touch_y_rel = global_pos_y - child.global_y
                
                # Check if the touch collides with the child
                if -256/2 <= touch_x_rel < 256/2 and -256/2 <= touch_y_rel < 256/2:
                    print("Clicked on child:", child)
                    print("Child Pos:", child.global_x, child.global_y)
                    print("Relative Touch Pos:", touch_x_rel, touch_y_rel)
                    print("Tile: ", child.x_tile, child.y_tile)
                    
                    # Convert relative touch coordinates to pixel coordinates within the tile
                    pixel_x = (touch_x_rel + 256 / 2)
                    pixel_y = (256 / 2 - touch_y_rel)
                    
                    # Convert tile coordinates to latitude and longitude
                    latitude, longitude = tile_pixel_to_lat_lon(child.zoom, child.x_tile, child.y_tile, pixel_x, pixel_y)
                    
                    print(f"Coordinates: {latitude}, {longitude}")
                    
                    # Perform reverse geocoding
                    if self.reverse_geocoding:
                        address = self.reverse_geocode(latitude, longitude)
                        #'amenity': 'hospital'
                        tags = {"amenity": "hospital"}
                        #tags = {"tourism": "hotel"}  # Example tags
                        places = self.query_osm_places_within_radius(latitude, longitude, 1000, tags)
                        #self.show_places_on_map(draggable_map, places)
                        for element in places["elements"]:
                            tags = element.get("tags", {})
                            name = tags.get("name", "Unnamed")
                            street = tags.get("addr:street", "")
                            lat = element.get("lat", "Unknown")
                            lon = element.get("lon", "Unknown")
                            print(f"{name} ({lat}, {lon}) - {street}")
                            
                            try:
                                self.add_circle(lat, lon, 16, 1)
                                
                            except:
                                pass
                        
                        # Filter out elements without 'lat' and 'lon' keys
                        valid_places = [p for p in places["elements"] if 'lat' in p and 'lon' in p]

                        if valid_places:
                            nearest_place = min(valid_places, key=lambda p: self.haversine_distance(latitude, longitude, p["lat"], p["lon"]))
                            tags = nearest_place.get("tags", {})
                            name = tags.get("name", "Unnamed")
                            street = tags.get("addr:street", "")
                            lat = nearest_place.get("lat", "Unknown")
                            lon = nearest_place.get("lon", "Unknown")
                            print(f"Nearest Place: {name} ({lat}, {lon}) - {street}")
                            
                            # Calculate distance using the haversine formula
                            distance_km = self.haversine_distance(latitude, longitude, lat, lon)
                            
                            # Print the distance
                            print(f"Nearest Place: {name} ({lat}, {lon}) - {street}")
                            print(f"Distance to nearest place: {distance_km:.2f} km")
                            
                            lines_arr_points = []
                            try:
                                self.add_circle(lat, lon, 16, 2)  # Add a circle at the nearest place location
                                route_coords, G, route = self.get_route_coords((latitude, longitude), (lat,lon))
                                # Define the projection from WGS 84 to Web Mercator (EPSG:3857)
                                
                                # Calculate total distance and time
                                total_distance_m = 0  # Total distance in meters
                                total_travel_time_sec = 0  # Total travel time in seconds

                                for u, v, key, edge_data in G.edges(keys=True, data=True):
                                    # Get the maxspeed for this edge, if available
                                    maxspeed = edge_data.get('maxspeed', 'Not available')
                                    
                                    # If 'maxspeed' is a list (which sometimes it is), print it as a string
                                    if isinstance(maxspeed, list):
                                        maxspeed = ', '.join(map(str, maxspeed))
                                    
                                    # Print the edge and its maxspeed
                                    print(f"Edge ({u}, {v}) - Maxspeed: {maxspeed}")
                                
                                print("Printing Route")
                                print(route)
                                
                                distance_km = self.total_route_distance(route_coords)
                                print(f"Total route distance: {distance_km:.2f} km")
                                # Calculate the travel time based on the route data
                                # Assuming `route` contains the total travel time (in seconds), otherwise, you will need to modify this
                                """
                                travel_time_sec = sum([edge['travel_time'] for edge in route])
                                travel_time_min = travel_time_sec / 60
                                
                                # Print the travel time
                                print(f"Estimated travel time: {travel_time_min:.2f} minutes")
                                """
                                for coords in route_coords:
                                    try:
                                        #x, y = self.wgs84_to_web_mercator(coords[0], coords[1])
                                        #lat_y, lon_x = self.web_mercator_to_degrees(x,y)
                                        
                                        #Plot path
                                        circle_pos = self.add_circle(coords[0], coords[1], 16, 3)
                                        lines_arr_points.append(circle_pos)
                                    except Exception as e:
                                        print(e)
                                la_len = len(lines_arr_points)
                                for i in range(1,la_len):
                                    self.add_path_line(lines_arr_points[i-1], lines_arr_points[i])
                            except Exception as e:
                                print(f"Error adding circle: {e}")

                        print(f"Address: {address}")

                        
                        self.add_circle(latitude, longitude, 16, 0)
                        
                        return child
                        
                        

        print("Global Pos: ", global_pos_x, global_pos_y, "\n")
        return super().on_touch_down(touch)
    def on_touch_move(self, touch):
        global global_offset
        global_offset[0] += touch.dx
        global_offset[1] += touch.dy
        
        for child in self.children:
            if isinstance(child, DraggableImage):
                child.x += touch.dx
                child.y += touch.dy
                
                #child.global_x = child.init_x + global_offset[0]
                #child.global_y = child.init_y - global_offset[1]
        self.update_circle_positions(touch.dx, touch.dy)  # Update circle positions
        self.update_path_line_positions(touch.dx, touch.dy)
        return super().on_touch_move(touch)
        
    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super().on_touch_up(touch)

    def add_tile(self, tile):
        self.add_widget(tile)
        self.tiles.append(tile)

    def remove_tile(self, tile):
        self.remove_widget(tile)
        self.tiles.remove(tile)

def download_tile_image(zoom, xtile, ytile, folder="tiles"):
    url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    filename = f"{folder}/{zoom}/{xtile}/{ytile}.png"
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"Tile image already exists: {filename}")
        return filename
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Define the custom user agent
    headers = {
        'User-Agent': 'my_app/1.0'  # Replace with your custom user agent
    }
    
    # Download the image with the custom user agent
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Tile image saved as: {filename}")
        return filename
    elif response.status_code == 418:
        print(f"Failed to download tile image: HTTP 418 - I'm a teapot")
    else:
        print(f"Failed to download tile image: HTTP {response.status_code} - {response.reason}")
    return filename

def tile_pixel_to_lat_lon(zoom, x_tile, y_tile, x_pixel, y_pixel):
    # Number of tiles at this zoom level
    n = 2.0 ** zoom
    
    # Normalized device coordinates
    x_n = (x_tile + (x_pixel / 256.0)) / n
    y_n = (y_tile + (y_pixel / 256.0)) / n
    
    # Longitude
    longitude = (x_n * 360.0) - 180.0
    
    # Latitude
    lat_rad = math.atan(math.sinh(math.pi * (1 - (2 * y_n))))
    latitude = math.degrees(lat_rad)
    
    return latitude, longitude
    
def lat_lon_to_tile_pixel(latitude, longitude, zoom):
    n = 2.0 ** zoom
    x_tile = int((longitude + 180.0) / 360.0 * n)
    lat_rad = math.radians(latitude)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x_tile, y_tile

def lat_lon_to_tile_pixel_with_pixel(latitude, longitude, zoom):
    # Calculate the number of tiles at the given zoom level
    n = 2.0 ** zoom
    
    # Calculate the tile coordinates
    x_tile = (longitude + 180.0) / 360.0 * n
    lat_rad = math.radians(latitude)
    y_tile = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    
    # Calculate the pixel coordinates within the tile
    x_pixel = (x_tile - int(x_tile)) * 256
    y_pixel = (y_tile - int(y_tile)) * 256
    
    # Convert tile coordinates to integers
    x_tile = int(x_tile)
    y_tile = int(y_tile)
    
    # Adjust pixel coordinates to be relative to the center of the tile
    x_pixel_centered = x_pixel - 128  # Assuming the tile size is 256
    y_pixel_centered = y_pixel - 128  # Assuming the tile size is 256
    
    return x_tile, y_tile, x_pixel_centered, y_pixel_centered

"""
# Define a wrapper function to run async functions from sync context
def async_wrapper(f, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(f(*args, **kwargs))
"""
TOGETHER_API_KEY = "4070baa3baed3400f79377ea3b4221f2024725f970bbf02dd0b6d4fba2175bc6"

class MapScreen(Screen):
    def __init__(self, **kwargs):
        super(MapScreen, self).__init__(**kwargs)
        # Create a FloatLayout to allow overlapping widgets
        self.float_layout = FloatLayout()

        # Declare map layout
        self.map_layout = BoxLayout(orientation='vertical', size_hint=(1, 1))
        self.draggable_map = DraggableMapScreen()
        self.map_layout.add_widget(self.draggable_map)

        
        # Define the range of tiles you want to load
        start_x = -5
        end_x = 5
        start_y = -5
        end_y = 5
        zoom = 16
        
        x_tile_init, y_tile_init = lat_lon_to_tile_pixel(48.8583, 2.2963, 16)

        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                x_tile, y_tile = (x_tile_init + x, y_tile_init + y)  # Adjust the starting tile numbers
                offset_x, offset_y = (x * 256, -y * 256)
                filename = download_tile_image(zoom, x_tile, y_tile)
                if filename:
                    new_image = DraggableImage(
                        source=filename, size=(256, 256), zoom=zoom, x_tile=x_tile, y_tile=y_tile,
                        allow_stretch=False, offset_x=offset_x, offset_y=offset_y
                    )
                    self.draggable_map.add_tile(new_image)
                    images_dict[(zoom, x_tile, y_tile)] = new_image
        
        # Toggle button for reverse geocoding
        toggle_button = ToggleButton(text='Reverse Geocoding Off', size_hint=(1, None), height=50)
        toggle_button.bind(on_press=lambda instance: self.toggle_reverse_geocoding(instance, self.draggable_map))
        self.map_layout.add_widget(toggle_button)

        # Add map_layout to float_layout
        self.float_layout.add_widget(self.map_layout)

        # Declare top layout and back button
        self.top_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, pos_hint={'top': 1})
        self.back_button = Button(text='Back', size_hint_x=None, width=100)
        self.back_button.bind(on_press=self.back_button_on_press)
        self.top_layout.add_widget(self.back_button)

        # Add top_layout to float_layout, it will be on top due to z-ordering
        self.float_layout.add_widget(self.top_layout)

        # Add float_layout to the screen
        self.add_widget(self.float_layout)
    
    def toggle_reverse_geocoding(self, instance, draggable_map):
        if instance.state == 'down':
            instance.text = 'Reverse Geocoding On: Searching Hospitals'
            draggable_map.reverse_geocoding = True
            instance.background_color = (0, 1, 0, 1)  # Green color
        else:
            instance.text = 'Reverse Geocoding Off'
            draggable_map.reverse_geocoding = False
            instance.background_color = (1, 0, 0, 1)  # Red color
            
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
                    async_nodes[i].trigger()
                    
                    # Your existing code here...
                except RecursionError:
                    print("Maximum recursion depth reached. Stopping program.")
                    # Additional cleanup or handling here if needed
        
        
    def on_run_press_wrapper(self, instance, node):
        def run_coroutine_in_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.on_run_press(node))

        threading.Thread(target=run_coroutine_in_event_loop).start()
        
    def back_button_on_press(self, instance):
        app = MDApp.get_running_app()
        self.manager.transition = NoTransition()
        self.manager.current = 'draggable_label_screen'
        
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

class NewAppComponent(BoxLayout):
    def __init__(self, text, **kwargs):
        super(NewAppComponent, self).__init__(orientation='horizontal', **kwargs)
        self.text = text
        
        self.label = Label(text=text, size_hint_x=0.7, halign='left', valign='middle')
        self.label.bind(size=self.label.setter('text_size'))  # Ensure the text size matches the label size
        
        self.button_box = BoxLayout(size_hint_x=0.3, orientation="horizontal")
        
        self.add_button = Button(text="Select")
        self.add_button.bind(on_press=self.button_on_press)
        
        self.button_box.add_widget(self.add_button)
        self.add_widget(self.label)
        self.add_widget(self.button_box)


    def button_on_press(self, instance):
        try:
            print(self.text)
            app = MDApp.get_running_app()
            app.root.current = apps[self.text]
            #app.manager.transition = NoTransition()
            #app.root.current = "draggable_label_screen"

        except Exception as e:
            print(e)

apps = {
    "Chatbot" : 'chatbox',
    "Renderer" : 'render_screen',
    "Map" : 'map_screen',
    "Lora Trainer" : 'lora_trainer_screen'
}

class LoraTrainerScreen(Screen):
    def __init__(self, **kwargs):
        super(LoraTrainerScreen, self).__init__(**kwargs)
        
        # Your Hugging Face token
        HF_TOKEN = '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'  # Replace with your actual Hugging Face token
        try:
        # Authenticate with Hugging Face
            login(token=HF_TOKEN)
        except:
            pass
        # Existing DataFrame
        self.new_data = pd.DataFrame({
            'prompt': []
        })
        
        self.data_points = []
        # Create the main layout for the screen
        screen_layout = BoxLayout(orientation='vertical')
        
        # Create the back button layout
        back_box = BoxLayout(size_hint=(1, None), height=40)
        back_button = Button(text="Back")
        back_button.bind(on_press=self.back_button_on_press)
        back_box.add_widget(back_button)
        
        # Create additional input boxes
        hugging_face_input_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        hugging_face_label = Label(text="HuggingFace Authtoken", size_hint_x=.25)
        self.hugging_face_input = TextInput(size_hint_x=.75, multiline=False)
        hugging_face_input_box.add_widget(hugging_face_label)
        hugging_face_input_box.add_widget(self.hugging_face_input)
        
        repo_id_input_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        repo_id_label = Label(text="repo_id", size_hint_x=.25)
        self.repo_id_input = TextInput(size_hint_x=.75, multiline=False)
        repo_id_input_box.add_widget(repo_id_label)
        repo_id_input_box.add_widget(self.repo_id_input)
        
        input_input_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=100)
        input_label = Label(text="Input", size_hint_x=.25)
        self.input_input = TextInput(size_hint_x=.75, multiline=True)
        input_input_box.add_widget(input_label)
        input_input_box.add_widget(self.input_input)
        
        instruction_input_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=100)
        instruction_label = Label(text="Instruction", size_hint_x=.25)
        self.instruction_input = TextInput(size_hint_x=.75, multiline=True)
        instruction_input_box.add_widget(instruction_label)
        instruction_input_box.add_widget(self.instruction_input)
        
        output_input_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=100)
        output_label = Label(text="Output", size_hint_x=.25)
        self.output_input = TextInput(size_hint_x=.75, multiline=True)
        output_input_box.add_widget(output_label)
        output_input_box.add_widget(self.output_input)
        
        save_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        
        add_to_batch_button = Button(text="Add To Batch")
        add_to_batch_button.bind(on_press=self.add_to_batch)
        save_box.add_widget(add_to_batch_button)
        
        append_batch_button = Button(text="Append Batch to Dataset")
        append_batch_button.bind(on_press=self.append_batch)
        save_box.add_widget(append_batch_button)
        
        upload_batch_button = Button(text="Upload Dataset to HuggingFace")
        upload_batch_button.bind(on_press=self.upload_batch)
        save_box.add_widget(upload_batch_button)
        
        log_box = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50)
        log_label_text = Label(text="Log", size_hint_x=0.75)
        reset_log_button = Button(text="Reset Log", size_hint_x=0.25)
        log_box.add_widget(log_label_text)
        log_box.add_widget(reset_log_button)
        
        
        self.log_textinput = TextInput(size_hint=(1, None), multiline=True, readonly=True, hint_text="Logs show here")
        

        # Add the back box and additional input boxes to the screen layout
        screen_layout.add_widget(back_box)  # Back box at the top
        screen_layout.add_widget(hugging_face_input_box)  # Additional inputs below
        screen_layout.add_widget(repo_id_input_box)
        screen_layout.add_widget(input_input_box)
        screen_layout.add_widget(instruction_input_box)
        screen_layout.add_widget(output_input_box)
        screen_layout.add_widget(save_box)
        screen_layout.add_widget(log_box)
        screen_layout.add_widget(self.log_textinput)
        
        # Add the screen layout to the screen
        self.add_widget(screen_layout)
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
                    async_nodes[i].trigger()
                    
                    # Your existing code here...
                except RecursionError:
                    print("Maximum recursion depth reached. Stopping program.")
                    # Additional cleanup or handling here if needed
        
    def on_run_press_wrapper(self, instance):
        def run_coroutine_in_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.on_run_press())
    def back_button_on_press(self, instance):
        app = App.get_running_app()
        self.manager.current = 'draggable_label_screen'
    # Define the generate_prompt function
    def add_to_batch(self, instance):
        data_point = {"input": self.input_input.text, "instruction": self.instruction_input.text, "output": self.output_input.text}
        self.data_points.append(data_point)
        self.log_textinput.text += "Added to 1 prompt batch\n"
    def append_batch(self, instance):
        # Generate prompts and create a new DataFrame
        generated_prompts = [self.generate_prompt(dp) for dp in self.data_points]
        generated_data = pd.DataFrame({'prompt': generated_prompts})

        # Append the new DataFrame to the existing one
        new_data = pd.concat([self.new_data, generated_data], ignore_index=True)

        #print(new_data)

        # Convert DataFrame to numpy array of bytes for text data
        new_data_array = new_data['prompt'].astype('S256').to_numpy()
        # File path
        hdf5_file = 'prompts.h5'
        self.append_data_to_hdf5(hdf5_file, new_data_array)
        self.data_points = []
        self.log_textinput.text += f"Appended batch to {hdf5_file}\n"
    def upload_batch(self, instance):
        try:
            # Hugging Face repository setup
            repo_id = self.repo_id_input.text or "IanVilla/gemma-2-finetuning"  # Replace with your Hugging Face username and desired repo name
            hdf5_file = 'prompts.h5'
            # Initialize Hugging Face API
            api = HfApi()
            HF_TOKEN = self.hugging_face_input.text
            login(token=HF_TOKEN)
        except Exception as e:
            print(e)

        # Create the repository if it does not exist
        try:
            # Check if repo exists
            repo_info = api.repo_info(repo_id)
            print(f"Repository '{repo_id}' already exists.")
        except Exception as e:
            # Create repo if it does not exist
            try:
                api.create_repo(repo_id, repo_type="dataset")
                print(f"Repository '{repo_id}' created.")
            except Exception as e:
                print(f"An error occurred while creating the repository: {e}")

        # Upload the file to Hugging Face (this will overwrite the existing file)
        try:
            upload_file(
                path_or_fileobj=hdf5_file,
                path_in_repo=hdf5_file,
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message="Updated HDF5 file with new data",
            )
            print(f"File '{hdf5_file}' updated in repo '{repo_id}'.")
        except Exception as e:
            print(f"An error occurred while uploading the file: {e}")

        df = self.read_hdf5('prompts.h5')
        print(df)

        # Convert the DataFrame to a Dataset object
        dataset = Dataset.from_pandas(df)

        # Create a DatasetDict object
        dataset_dict = DatasetDict({
            "train": dataset
        })

        # Upload the dataset to Hugging Face
        dataset_dict.push_to_hub(repo_id=repo_id, token=HF_TOKEN)
    def generate_prompt(self, data_point):
        """Generate input text based on a prompt, task instruction, (context info.), and answer

        :param data_point: dict: Data point
        :return: str: tokenized prompt
        """
        prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                       'appropriately completes the request.\n\n'
        if data_point['input']:
            text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
        else:
            text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
        return text
    def read_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            if 'prompts' in f:
                # Access the dataset
                dataset = f['prompts']
                # Convert to DataFrame
                # Since the data is stored as bytes, decode and convert to DataFrame
                data = [entry.decode('utf-8') for entry in dataset[:]]
                df = pd.DataFrame(data, columns=['prompt'])
                return df
            else:
                print("Dataset 'prompts' not found.")
                return None


    # Function to append data to HDF5
    def append_data_to_hdf5(self, file_path, new_data):
        with h5py.File(file_path, 'a') as f:
            if 'prompts' in f:
                # Dataset exists, extend it
                dataset = f['prompts']
                # Resize dataset to accommodate new data
                old_size = dataset.shape[0]
                new_size = old_size + new_data.shape[0]
                dataset.resize(new_size, axis=0)
                # Append new data
                dataset[old_size:] = new_data
            else:
                # Dataset does not exist, create it
                f.create_dataset('prompts', data=new_data, maxshape=(None,), dtype='S256')


class SelectAppScreen(Screen):
    def __init__(self, **kwargs):
        super(SelectAppScreen, self).__init__(**kwargs)
        
        # Create the main layout for the screen
        screen_layout = BoxLayout(orientation='vertical')
        
        # Create the back button layout
        back_box = BoxLayout(size_hint=(1, None), height=40)
        
        back_button = Button(text="Back")
        back_button.bind(on_press=self.back_button_on_press)
        
        refresh_button = Button(text="Refresh")
        refresh_button.bind(on_press=self.refresh_components)
        back_box.add_widget(back_button)
        back_box.add_widget(refresh_button)
        
        search_box = BoxLayout(size_hint=(1, None), height=40)
        
        self.search_input = TextInput(size_hint_x = .75, multiline=False)
        search_button = Button(text="Search", size_hint_x = .25)
        search_box.add_widget(self.search_input)
        search_box.add_widget(search_button)
        # Create the main scroll view
        main_scroll = ScrollView(size_hint=(1, 1))
        
        # Create the main layout inside the scroll view
        self.main_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.main_layout.bind(minimum_height=self.main_layout.setter('height'))
        
        self.add_custom_components()
        
        # Add the main layout to the scroll view
        main_scroll.add_widget(self.main_layout)
        
        # Add the back button and scroll view to the screen layout
        screen_layout.add_widget(back_box)
        screen_layout.add_widget(search_box)
        screen_layout.add_widget(main_scroll)
        
        # Add the screen layout to the screen
        self.add_widget(screen_layout)
        
        #self.clear_custom_components()
    def search_nodes(self, instance):
        for i in node_init:
            pass
            
    def refresh_components(self, instance):
        self.clear_custom_components()
        self.add_custom_components()
        
    def clear_custom_components(self):
        # Clear all children from the main layout
        print("Cleared!")
        self.main_layout.clear_widgets()
        
    def add_custom_components(self):
        # Add custom components to the main layout
        print("Added!")
        for i in apps:  # Adding multiple custom components
            print(i)
            custom_component = NewAppComponent(text=f"{i}")
            custom_component.size_hint_y = None
            custom_component.height = 50
            self.main_layout.add_widget(custom_component)
    def back_button_on_press(self, instance):
        app = MDApp.get_running_app()
        self.manager.transition = NoTransition()
        self.manager.current = 'draggable_label_screen'

'''
Apps List
apps = {
    "Chatbot" : #name of the screen
    "Renderer" : #name of the screen
    "Map" : #name of the screen
}

Agents List
agent_screens = {
    #Agent name
    #Agent screen name
    
    #Send to agent node
    #Recieve from Agent node
    
    #Then like each agent can send inputs to other agents
    #Example agent can choose which to run. Like an instruct type, the agent function, and the inputs.
}
'''

class SelectNodeScreen(Screen):
    def __init__(self, **kwargs):
        super(SelectNodeScreen, self).__init__(**kwargs)
        
        # Create the main layout for the screen
        screen_layout = BoxLayout(orientation='vertical')
        
        # Create the back button layout
        back_box = BoxLayout(size_hint=(1, None), height=40)
        
        back_button = Button(text="Back")
        back_button.bind(on_press=self.back_button_on_press)
        
        refresh_button = Button(text="Refresh")
        refresh_button.bind(on_press=self.refresh_components)
        back_box.add_widget(back_button)
        back_box.add_widget(refresh_button)
        
        search_box = BoxLayout(size_hint=(1, None), height=40)
        
        self.search_input = TextInput(size_hint_x = .75, multiline=False)
        search_button = Button(text="Search", size_hint_x = .25)
        search_box.add_widget(self.search_input)
        search_box.add_widget(search_button)
        # Create the main scroll view
        main_scroll = ScrollView(size_hint=(1, 1))
        
        # Create the main layout inside the scroll view
        self.main_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.main_layout.bind(minimum_height=self.main_layout.setter('height'))
        
        self.add_custom_components()
        
        # Add the main layout to the scroll view
        main_scroll.add_widget(self.main_layout)
        
        # Add the back button and scroll view to the screen layout
        screen_layout.add_widget(back_box)
        screen_layout.add_widget(search_box)
        screen_layout.add_widget(main_scroll)
        
        # Add the screen layout to the screen
        self.add_widget(screen_layout)
        
        #self.clear_custom_components()
    def search_nodes(self, instance):
        for i in node_init:
            pass
            
    def refresh_components(self, instance):
        self.clear_custom_components()
        self.add_custom_components()
        
    def clear_custom_components(self):
        # Clear all children from the main layout
        print("Cleared!")
        self.main_layout.clear_widgets()
        
    def add_custom_components(self):
        # Add custom components to the main layout
        print("Added!")
        for i in node_init:  # Adding multiple custom components
            print(i)
            custom_component = NewNodeComponent(text=f"{i}")
            custom_component.size_hint_y = None
            custom_component.height = 50
            self.main_layout.add_widget(custom_component)
    
    def fetch_and_process_query(user_query):
        url = "https://api.vectara.io/v2/query"
        
        payload = json.dumps({
            "query": f"{user_query}",
            "search": {
                "corpora": [
                    {
                        "custom_dimensions": {},
                        "metadata_filter": "doc.date_downloaded<'2024-06-18'",
                        "lexical_interpolation": 0.025,
                        "semantics": "default",
                        "corpus_key": "Semantic_Search_2"
                    }
                ],
                "offset": 0,
                "limit": 5,
                "context_configuration": {
                    "characters_before": 30,
                    "characters_after": 30,
                    "sentences_before": 3,
                    "sentences_after": 3,
                    "start_tag": "<em>",
                    "end_tag": "</em>"
                },
            },
            "stream_response": False
        })

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'customer-id': '4239971555',
            'x-api-key': 'zwt__LjU4zVE9TBF1tU4SbJ0rrjT1QWwI2sXXp4iGQ'
        }

        # Make the request
        response = requests.request("POST", url, headers=headers, data=payload)
        
        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")

        # Save the JSON response to a file
        with open('response.json', 'w') as f:
            json.dump(response.json(), f, indent=4)

        # Load the JSON response from the file
        with open('response.json', 'r') as f:
            json_result = json.load(f)

        # Extract and print the context
        context = ""
        for result in json_result['search_results']:
            context += f"Text: {result['text']}\n\n"
            print(f"Text: {result['text']}\n\n")
        
        return context
        # Example usage
        """
        user_query = "sarin gas"
        context = fetch_and_process_query(user_query)
        print(context)
        """
    def send_one_node_to_vectara(self, json_data, metadata_args, file_path):
        metadata = {
            #"metadata_key": "metadata_value",
            "date_downloaded": metadata_args["date_downloaded"],
            "date_uploaded": re.sub(r'[/:.]', '_', datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
            "epoch" : int(time.time()),
            "node_name" : metadata_args["node_name"],
            "filename" : metadata_args["filename"],
        }
        json_data_bytes = json.dumps(json_data).encode('utf-8')
        url = "https://api.vectara.io/v1/upload?c=4239971555&o=2"
        headers = {
            'x-api-key': 'zwt__LjU4zVE9TBF1tU4SbJ0rrjT1QWwI2sXXp4iGQ'
        }
        now = datetime.utcnow().strftime("%Y_%m_%d %H_%M_%S UTC")
        # Save the JSON-formatted string to a file
        with open(f"pages_json/{metadata['filename']}.json", "w") as file:
            file.write(json.dumps(json_data))
        files = {
            "file": (f"{metadata['filename']}", json_data_bytes, 'rb'),
            "doc_metadata": (None, json.dumps(metadata)),  # Replace with your metadata
        }
        response = requests.post(url, headers=headers, files=files)
        print(response.text)
    def delete_node(self, corpus_key, document_id):
        url = f"https://api.vectara.io/v2/corpora/{corpus_key}/documents/{document_id}"
        #url = "https://api.vectara.io/v2/corpora/Semantic_Search_2/documents/2024-04-18%2013_21_57%20UTC|https___unstructured-io_github_io_unstructured_core_partition_html"
        
        payload={}
        headers = {
          'x-api-key': 'zwt__LjU4zVE9TBF1tU4SbJ0rrjT1QWwI2sXXp4iGQ'
        }

        response = requests.request("DELETE", url, headers=headers, data=payload)

        print(response.text)
    
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
            try:
                exec(node_init[i]["function_string"], globals())
                exec(formatted_string, globals())
            except Exception as e:
                print(e)
                pass
            
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
        top_layout = BoxLayout(orientation='vertical', size_hint=(1, None), height=80, pos_hint={'top' : 1})
        
        apps_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        agents_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40)
        
        agent_button = Button(text='Agents')
        apps_button = Button(text='Apps', on_press=self.switch_to_apps)
        
        agents_layout.add_widget(agent_button)
        apps_layout.add_widget(apps_button)
        
        chatbot_button = Button(text='Chatbot')
        apps_layout.add_widget(chatbot_button)
        
        render_button = Button(text='Renderer', on_press=self.switch_to_renderer)
        apps_layout.add_widget(render_button)
        
        kuka_button = Button(text='Kuka', on_press=self.switch_to_kuka)
        apps_layout.add_widget(kuka_button)
        
        # Bind button press to switch_to_screen method
        chatbot_button.bind(on_press=self.switch_to_screen)
        
        root.add_widget(top_layout)
        
        top_layout.add_widget(apps_layout)
        top_layout.add_widget(agents_layout)
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
                    async_nodes[i].trigger()
                    
                    # Your existing code here...
                except RecursionError:
                    print("Maximum recursion depth reached. Stopping program.")
                    # Additional cleanup or handling here if needed
        #await asyncio.gather(*tasks)
        
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
        
    def switch_to_kuka(self, instance):
        # Switch to 'chatbox'
        self.manager.transition = NoTransition()
        self.manager.current = 'kuka_screen'
    
    def switch_to_apps(self, instance):
        self.manager.transition = NoTransition()
        self.manager.current = 'select_app_screen'

# Define global variables for position and gripper position
pos = [-0.4, 0.0, 0.5]
gripper_pos = 0.0  # Initial value for gripper position
current_pos = [0,0,0]
ball_pos = [0,0,0]
target_pos = None
ball_color = "ball_1"

class KukaScreen(Screen):
    x_pos = NumericProperty(-0.4)
    y_pos = NumericProperty(0.0)
    z_pos = NumericProperty(0.5)
    gripper_pos = NumericProperty(0.0)
    roll_angle = NumericProperty(0.0)  # Add a property for roll angle

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_image = False
        self.gripped = False
        self.track_ball = False  # Flag to control tracking the ball
        self.new_transition = False
        
        # Create the main BoxLayout container
        main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        # Create the back button layout
        back_box = BoxLayout(size_hint=(1, None), height=40)
        back_button = Button(text="Back", on_press=self.switch_to_screen)
        main_layout.add_widget(back_button)
        # Create a BoxLayout for sliders and labels
        slider_layout = BoxLayout(orientation='vertical', spacing=5)
        
        # Label and slider for x position
        slider_layout.add_widget(Label(text="X Position"))
        self.x_slider = Slider(min=-2, max=2, value=self.x_pos)
        self.x_slider.bind(value=self.update_x_pos)
        slider_layout.add_widget(self.x_slider)

        # Label and slider for y position
        slider_layout.add_widget(Label(text="Y Position"))
        self.y_slider = Slider(min=-2, max=2, value=self.y_pos)
        self.y_slider.bind(value=self.update_y_pos)
        slider_layout.add_widget(self.y_slider)

        # Label and slider for z position
        slider_layout.add_widget(Label(text="Z Position"))
        self.z_slider = Slider(min=0, max=2, value=self.z_pos)
        self.z_slider.bind(value=self.update_z_pos)
        slider_layout.add_widget(self.z_slider)

        # Label and slider for gripper position
        slider_layout.add_widget(Label(text="Gripper Position"))
        self.gripper_slider = Slider(min=0, max=0.8, value=self.gripper_pos)
        self.gripper_slider.bind(value=self.update_gripper_pos)
        slider_layout.add_widget(self.gripper_slider)

        # Label and slider for roll angle
        slider_layout.add_widget(Label(text="Roll Angle"))
        self.roll_slider = Slider(min=-math.pi, max=math.pi, value=self.roll_angle)  # Range for roll angle
        self.roll_slider.bind(value=self.update_roll_angle)
        slider_layout.add_widget(self.roll_slider)

        # Add the slider layout to the main layout
        main_layout.add_widget(slider_layout)

        # Text input for controlling the gripper
        self.command_input = TextInput(
            hint_text="Enter command (e.g., open, close)", 
            multiline=False,
            height=50,  # Set height
            size_hint=(1, None)  # Allow it to scale with width but not height
        )
        self.command_input.bind(on_text_validate=self.on_command_entered)
        main_layout.add_widget(self.command_input)
        
        # Add a button to open the file selector
        file_button = Button(text="Open File", size_hint=(1, None), height=40)
        file_button.bind(on_press=self.open_image_file_selector)
        main_layout.add_widget(file_button)
        
        # Add an Image widget for displaying the camera view
        self.camera_view = Image(size_hint=(1, None), height=400)  # Make it larger
        main_layout.add_widget(self.camera_view)
        
        # Set the layout of the screen to the main layout
        self.add_widget(main_layout)
     
    def open_image_file_selector(self, instance):
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]  # Limit to image files
        )
        if file_path:
            print(f"Selected file: {file_path}")  # Do something with the selected image file
            self.selected_image = file_path
            self.process_image()
            
    def switch_to_screen(self, instance):
        # Switch to 'chatbox'
        self.manager.transition = NoTransition()
        self.manager.current = 'draggable_label_screen'
        
    def update_x_pos(self, instance, value):
        global pos
        pos[0] = value

    def update_y_pos(self, instance, value):
        global pos
        pos[1] = value

    def update_z_pos(self, instance, value):
        global pos
        pos[2] = value

    def update_gripper_pos(self, instance, value):
        global gripper_pos
        gripper_pos = value  # Update the gripper position based on the slider value

    def update_roll_angle(self, instance, value):
        self.roll_angle = value  # Update the roll angle based on the slider value

    def update_camera_view(self, image_data):
        if image_data is not None:
            # Schedule the UI update to run on the main thread
            Clock.schedule_once(lambda dt: self._update_texture(image_data))

    def _update_texture(self, image_data):
        # Internal method to update the Image widget with new camera data
        texture = Texture.create(size=(image_data.shape[1], image_data.shape[0]), colorfmt='rgb')
        texture.blit_buffer(image_data.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        self.camera_view.texture = texture
        
    def on_command_entered(self, instance=None):
        # Run the command handling in a separate thread to avoid blocking the UI
        threading.Thread(target=self.process_command, args=(self.command_input.text.strip(),)).start()
        
    def process_image(self):
        global public_url
        # Use regex to extract the Ngrok public URL and the local path
        match = re.search(r'"(https?://[^\s]+)" -> "(http://localhost[^\s]+)"', str(public_url))
        ngrok_public_url = None
        #We need to move the image to the uploads folder
        
        if match:
            ngrok_public_url = match.group(1)  # Extracts the Ngrok public URL
            # Split by '/' and get the last part
            filename = self.selected_image.split('/')[-1]
            # Construct the URL for the uploaded file
            # Copy the file to the destination directory
            shutil.copy(self.selected_image, f"uploads/{filename}")
            file_url = f"{ngrok_public_url}/uploads/{filename}"
            print(file_url)
        
        if file_url:
            # Create the message array to be sent to the LLM
            message_array = [
                {
                    "role": "system",
                    "content": "Your role is to describe images comprehensively, including its colors, etc."
                },
                {
                    "role": "user",
                    "content": [
                            {
                                    "type": "text",
                                    "text": "What is in the image?"
                            },
                            {
                                    "type": "image_url",
                                    "image_url": {
                                            "url": file_url
                                    }
                            }
                    ]
                }
            ]
            
            
            # Interact with the LLM (this can take time, so we run it in a separate thread)
            chat_completion = client.chat.completions.create(
                messages=message_array,
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                max_tokens=5000
            )
            # Extract the response content
            response = chat_completion.choices[0].message.content
            print(response)
            
            self.process_command(command=response)
    
    def process_command(self, command):
        # For other commands, set the flag to False to stop tracking the ball
        self.track_ball = False
        self.gripped = False
        gripper_pos = 0  # Fully open the gripper
        print(f"Tracking stopped. Command: {command}")
        
        # Define the prompt to instruct the LLM on what to do
        generate_code = (
            f"User Input: {command}\n"
            "Command Types:\n"
            "0: Open the gripper.\n"
            "1: Close the gripper.\n"
            "2: Move north.\n"
            "3: Move south.\n"
            "4: Move east.\n"
            "5: Move west.\n"
            "6: Move up.\n"
            "7: Move down.\n"
            "8: Move to the ball.\n"
            "9: When none of the above\n"
            "Based on the input, return only the command number.\n"
            "Format: command type:<number>"
        )
        
        # Create the message array to be sent to the LLM
        message_array = [
            {"role": "system", "content": "Your role is to decide what the command type to use based on the user input. You will move to what the user describes"},
            {"role": "user", "content": generate_code}
        ]

        client_for_instruct = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url='https://api.together.xyz/v1',
        )
        
        # Interact with the LLM (this can take time, so we run it in a separate thread)
        chat_completion = client_for_instruct.chat.completions.create(
            messages=message_array,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )

        # Extract the response content
        response = chat_completion.choices[0].message.content

        # Use a regular expression to find the command type number in the response
        response = response.lower()  # Convert to lowercase for consistency
        pattern = re.compile(r"command type\s*:\s*(\d+)")
        match = pattern.search(response)
        command_type = int(match.group(1)) if match else None

        if command_type is None:
            print("command_type not found in the data")

        print("Bot Response:", response)

        # Handle the command execution in the main thread to update the UI
        Clock.schedule_once(lambda dt: self.execute_command(command_type, command), 0)

    def execute_command(self, command_type, command):
        # Execute action based on the command type
        global gripper_pos, target_pos, ball_pos
        if command_type == 0:
            gripper_pos = 0  # Fully open the gripper
        elif command_type == 1:
            gripper_pos = 0.8  # Fully close the gripper
        elif command_type == 2:
            target_pos = [0, 1, 0]  # Move north
        elif command_type == 3:
            target_pos = [0, -1, 0]  # Move south
        elif command_type == 4:
            target_pos = [1, 0, 0]  # Move east
        elif command_type == 5:
            target_pos = [-1, 0, 0]  # Move west
        elif command_type == 6:
            target_pos = [0, 0, 1]  # Move up
        elif command_type == 7:
            target_pos = [0, 0, -1]  # Move down
        elif command_type == 8:
            # Set track_ball to True to start tracking the ball
            # Set track_ball to True to start tracking the ball
            
            #Also change the color of the ball
            # Create the message array to be sent to the LLM
            # Define the prompt to instruct the LLM on what to do
            generate_code = (
                f"User Input: {command}\n"
                "Ball colors:\n"
                "0: Red.\n"
                "1: Green.\n"
                "2: Blue.\n"
                "3: Yellow.\n"
                "4: Magenta.\n"
                "Based on the input, return only the command number.\n"
                "Format: command type:<number>"
            )
            
            message_array = [
                {"role": "system", "content": "Your role is to decide what color of the ball is gonna get grabbed."},
                {"role": "user", "content": generate_code}
            ]
            
            client_for_instruct = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url='https://api.together.xyz/v1',
            )
            # Interact with the LLM
            chat_completion = client_for_instruct.chat.completions.create(
                messages=message_array,
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
            )

            # Extract the response content
            response = chat_completion.choices[0].message.content

            # Use a regular expression to find the command type number in the response
            response = response.lower()  # Convert to lowercase for consistency
            pattern = re.compile(r"command type\s*:\s*(\d+)")
            match = pattern.search(response)
            command_type = int(match.group(1)) if match else None
            global ball_color
            ball_color = f"ball_{command_type+1}"
            print("Ball color ", ball_color)
            self.track_ball = True
            print("Tracking Ball Command Received.")
        elif command_type == 9:
            print("Command Received: None of the above")

        # Optional: You can add more UI feedback or actions here
        self.new_transition = True
        # Start smooth transition (even if track_ball is True or False)
        self.start_smooth_transition()

        self.command_input.text = ""  # Clear the text input
        
    def start_smooth_transition(self):
        # Start updating the position smoothly
        self.transition_step = 0
        self.transition_max_steps = 60  # Total steps for the smooth transition (adjust for speed)
        print("Transitioning")
        Clock.schedule_interval(self.smooth_transition_update, 1/60)  # 60 FPS update

    def smooth_transition_update(self, dt):
        global current_pos, target_pos, pos, ball_pos, gripper_pos
        
        if self.track_ball:
            self.transition_step = self.transition_max_steps
            # Calculate the horizontal distance (X and Y distance) between the ball and the current position
            horizontal_distance = ((current_pos[0] - ball_pos[0])**2 + (current_pos[1] - ball_pos[1])**2)**0.5
            
            if horizontal_distance < 0.1 and not self.gripped:  # If the object is directly above the ball (within a small threshold)
                if not self.gripped:
                    target_pos = [ball_pos[0], ball_pos[1], ball_pos[2] + 0.45]  # Slightly above the ball
                    # Schedule the gripper to close after a short delay (e.g., 1 second)
                    pos = target_pos
                    Clock.schedule_once(self.close_gripper, 1)
                if self.gripped:
                    target_pos = [ball_pos[0], ball_pos[1], ball_pos[2] + 1]  # Slightly above the ball
            else:
                target_pos = [ball_pos[0], ball_pos[1], ball_pos[2] + 1]  # Higher above if not directly above
                pos = target_pos
                
            #print("Ball position being tracked:", target_pos)

        # Interpolate between current_pos and target_pos
        elif self.transition_step < self.transition_max_steps:
            if target_pos:
                t = self.transition_step / self.transition_max_steps
                current_pos = [
                    (1 - t) * current_pos[i] + t * target_pos[i] for i in range(3)
                ]
                pos = current_pos
                self.transition_step += 1
        else:
            if not self.new_transition:
                self.new_transition = False
                current_pos = target_pos  # Ensure it ends exactly on the target
                Clock.unschedule(self.smooth_transition_update)  # Stop the update

    def close_grip(self, dt):
        self.gripped = True
        
    def close_gripper(self, dt):
        global gripper_pos
        gripper_pos = 0.8  # Fully close the gripper
        # Schedule the function to close the grip after 2 seconds
        Clock.schedule_once(self.close_grip, 2)  # Delay closure of the grip by 2 seconds

def create_ball(position, radius=0.1, mass=1, color=[1, 0, 0, 1], friction_values=None):
    """
    Create a ball in the simulation at the specified position with the given radius, mass, and color.

    :param position: List of [x, y, z] coordinates for the ball's starting position.
    :param radius: Radius of the ball (default: 0.1).
    :param mass: Mass of the ball (default: 0.5).
    :param color: RGBA color list for the ball (default: [1, 0, 0, 1] for red).
    :param friction_values: Dictionary specifying friction properties (default: None).
                            Example: {"lateral": 5.0, "spinning": 1.5, "rolling": 1.0}
    :return: The unique ID of the created ball.
    """
    # Create collision and visual shapes for the ball
    ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    
    # Create the ball body
    ball_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=ball_collision_shape,
        baseVisualShapeIndex=ball_visual_shape,
        basePosition=position
    )
    
    
    # Apply friction if specified
    if friction_values:
        p.changeDynamics(
            ball_id, -1,
            lateralFriction=friction_values.get("lateral", 1e5),
            spinningFriction=friction_values.get("spinning", 1e5),
            rollingFriction=friction_values.get("rolling", 1e5)
        )
    else:
        p.changeDynamics(
            ball_id, -1,
            lateralFriction=1e5,
            spinningFriction=1e5,
            rollingFriction=1e5
        )
    
    # Reset velocity for a clean start
    p.resetBaseVelocity(ball_id, [0, 0, 0], [0, 0, 0])

    # Optionally apply gravity (if required by your setup)
    p.applyExternalForce(ball_id, -1, forceObj=[0, 0, -9.81], posObj=[0, 0, 0], flags=p.WORLD_FRAME)
    
    return ball_id

def start_pybullet_simulation(app_instance):
    global pos, gripper_pos, current_pos, ball_pos, ball_color
    clid = p.connect(p.SHARED_MEMORY)
    if clid < 0:
        p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, -0.3])
    kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
    kukaEndEffectorIndex = 6
    numJoints = p.getNumJoints(kukaId)
    ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]

    for i in range(numJoints):
        p.resetJointState(kukaId, i, rp[i])
    
    # Load the gripper URDF model
    gripper_id = p.loadURDF("gripper.urdf", basePosition=[0, 0, 0.5])

    num_joints = p.getNumJoints(gripper_id)
    print(f"Number of joints: {num_joints}")

    # Define the joint indices for the left and right fingers
    left_finger_joint = 0
    right_finger_joint = 1

    # Initial position for the fingers (open gripper)
    left_finger_position = 0.2
    right_finger_position = -0.2

    # Move the joints to the desired positions (open gripper)
    p.setJointMotorControl2(gripper_id, left_finger_joint, p.POSITION_CONTROL, targetPosition=left_finger_position)
    p.setJointMotorControl2(gripper_id, right_finger_joint, p.POSITION_CONTROL, targetPosition=right_finger_position)

    # Set gravity in the Z-direction (earth gravity)
    p.setGravity(0, 0, -9.81)  # Gravity in the negative Z-direction (downward)

    # Use discrete simulation steps
    p.setRealTimeSimulation(0)

    width, height = 320, 240
    fov = 60
    aspect = width / height
    near, far = 0.02, 5

    # Define the view matrix (camera position and orientation)
    camera_position = [0, 0, 2]  # Set a fixed position for the camera
    camera_target = [0, 0, 0]  # Look at the origin
    camera_up = [0, 0, 1]  # Camera's up direction
    viewMatrix = p.computeViewMatrix(camera_position, camera_target, camera_up)

    # Define the projection matrix
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    
    """
    # Create a ball at position (0.5, 0.5, 1) to give it height above the ground
    ball_radius = 0.1  # Adjust the ball size if needed
    # Create the ball collision and visual shapes
    ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1])  # Red color

    # Create the ball with the correct arguments
    ball_id = p.createMultiBody(
        baseMass=0.5, 
        baseCollisionShapeIndex=ball_collision_shape,
        baseVisualShapeIndex=ball_visual_shape, 
        basePosition=[0.5, 0, 1]  # Position the ball slightly above the ground
    )

    # Try higher friction values to increase resistance
    p.changeDynamics(ball_id, -1, lateralFriction=5.0, spinningFriction=1.5, rollingFriction=1.0)

    # Apply a small initial velocity to make sure the ball starts falling
    p.resetBaseVelocity(ball_id, [0, 0, 0], [0, 0, 0])  # Reset velocity to zero for a clean start

    # Ensure gravity is properly set by applying a force (as a test)
    p.applyExternalForce(ball_id, -1, forceObj=[0, 0, -9.81], posObj=[0, 0, 0], flags=p.WORLD_FRAME)
    """
    # Dictionary to store ball IDs with descriptive keys
    balls = {}

    # Number of balls to place in the circle
    num_balls = 5
    radius = 0.5
    center_position = [0, 0, 1]  # Center of the circle

    # Colors for the balls (adjust or add more as needed)
    colors = [
        [1, 0, 0, 1],  # Red
        [0, 1, 0, 1],  # Green
        [0, 0, 1, 1],  # Blue
        [1, 1, 0, 1],  # Yellow
        [1, 0, 1, 1]   # Magenta
    ]

    # Create balls arranged in a circle
    for i in range(num_balls):
        # Calculate the angle for this ball
        angle = (2 * math.pi / num_balls) * i  # Divide the circle into equal parts
        
        # Calculate x and y positions using trigonometry
        x = center_position[0] + radius * math.cos(angle)
        y = center_position[1] + radius * math.sin(angle)
        z = center_position[2]  # Keep the z-coordinate the same as the center
        
        # Create the ball and store it in the dictionary
        ball_key = f"ball_{i+1}"  # Generate a unique key for each ball
        balls[ball_key] = create_ball(position=[x, y, z], color=colors[i % len(colors)])
    
    while True:
        p.stepSimulation()
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul, jr, rp)
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], force=500)
        
        # Control the gripper (fingers) based on the gripper_pos slider value
        gripper_offset = gripper_pos * 0.2  # Max distance the fingers move
        grip_force = 5000  # Increase the force if needed
        p.setJointMotorControl2(gripper_id, left_finger_joint, p.POSITION_CONTROL, 
                                targetPosition=gripper_offset, force=grip_force)
        p.setJointMotorControl2(gripper_id, right_finger_joint, p.POSITION_CONTROL, 
                                targetPosition=-gripper_offset, force=grip_force)
        # Get the position of the ball
        ball_pos, ball_orientation = p.getBasePositionAndOrientation(balls[ball_color])
        
        # Print the position of the ball
        #print("Ball Position: ", ball_position)

        link_state = p.getLinkState(kukaId, kukaEndEffectorIndex)
        camera_position = link_state[0]
        
        # Get the position and orientation of the KUKA robot's end effector
        link_state = p.getLinkState(kukaId, kukaEndEffectorIndex)
        kuka_position = link_state[0]  # The position of the end effector (x, y, z)
        kuka_orientation = link_state[1]  # The orientation of the end effector (quaternion)

        # Define the pitch rotation (rotation around the X-axis)
        pitch_rotation = -math.pi / 2  # -90 degrees in radians
        pitch_quaternion = p.getQuaternionFromEuler([pitch_rotation, 0, 0])

        # Define the roll rotation (rotation around the Z-axis)
        roll_rotation = app_instance.roll_angle  # Get the roll angle from the slider
        roll_quaternion = p.getQuaternionFromEuler([0, 0, roll_rotation])

        # Multiply the original KUKA orientation by the pitch and roll rotation quaternions
        _, new_orientation = p.multiplyTransforms([0, 0, 0], kuka_orientation, [0, 0, 0], roll_quaternion)
        _, new_orientation = p.multiplyTransforms([0, 0, 0], new_orientation, [0, 0, 0], pitch_quaternion)

        # Set the gripper's position to the KUKA end effector's position
        p.resetBasePositionAndOrientation(gripper_id, kuka_position, new_orientation)
        current_pos = kuka_position
        view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_position, distance=0.5, yaw=0, pitch=-90, roll=0, upAxisIndex=2)
        projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
        
        # Capture camera image (RGB only, no depth or segmentation)
        img_arr = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Only process RGB data (ignore depth and segmentation)
        rgb_array = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]  # RGB data only

        # Update the camera view in the app
        app_instance.update_camera_view(rgb_array)

        # Get the camera's forward direction from the view matrix (this will be the camera's "look-at" direction)
        forward_vector = np.array([view_matrix[8], view_matrix[9], view_matrix[10]])
        time.sleep(1 / 240.)
        
class DraggableLabelApp(MDApp):
    past_messages = {}
    def build(self):
        self.theme_cls.theme_style = 'Dark'
        return Builder.load_string(KV)
    def on_start(self):
        # Code here runs right after the app has finished building
        print("Hello from on_start")
        """
        simulation_thread = threading.Thread(target=start_pybullet_simulation, args=(self.root.get_screen("kuka_screen"),))
        simulation_thread.daemon = True
        simulation_thread.start()
        """
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
    
    # Function to add a message to the list
    def add_message(self, isVisionModel, role, user_text=None, image=None, user_id=None, conversation_id=None):
        if role == "assistant":
            self.past_messages[user_id][conversation_id].append({
                "role": role,
                "content": user_text
            })
        if isVisionModel:
            if user_id not in self.past_messages:
                self.past_messages[user_id] = {}
            if conversation_id not in self.past_messages[user_id]:
                self.past_messages[user_id][conversation_id] = []
            self.past_messages[user_id][conversation_id].append({
                "role": "system",
                "content": "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them."
            })
            if image != None:
                self.past_messages[user_id][conversation_id].append({"role": role, "content": [{"type" : "text", "text" : user_text}, {"type" : "image_url", "image_url" : {"url" : image}}]})
            else:
                self.past_messages[user_id][conversation_id].append({"role": role, "content": [{"type" : "text", "text" : user_text}]})
        else:
            #O1 does not support system
            if user_id not in self.past_messages:
                self.past_messages[user_id] = {}
            if conversation_id not in self.past_messages[user_id]:
                self.past_messages[user_id][conversation_id] = []
            self.past_messages[user_id][conversation_id].append({
                "role": "system",
                "content": "Your role is to assist users by providing information, answering questions, and engaging in conversations on various topics. Whether users need help with programming, want to discuss philosophical questions, or just need someone to chat with, I'm here to assist them."
            })
                
            self.past_messages[user_id][conversation_id].append({"role": role, "content": user_text})
        
        if role == "user":
            print(f"User: {user_text}")
    # Function to continue the conversation
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    def continue_conversation(self, model=None, user_text=None, context=None, user_id=None, image=None, conversation_id=None):
        #print(past_messages)
        # Create the chat completion request with updated past messages
        # List of common image extensions
        
        # Check if filename ends with one of the image extensions
        if image:
            if not image.lower().endswith(self.image_extensions):
                image = None
        
        if user_text != None:
            model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"#"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" #"meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
            isVisionModel = True if model == "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo" else False
            self.add_message(isVisionModel, "user", user_text = user_text, image = image, user_id = user_id, conversation_id=conversation_id)
            if context:
                print("Adding context: ", context)
                self.add_message(isVisionModel, "user", user_text = user_text, image = image, user_id = user_id, conversation_id=conversation_id)
            
            chat_completion = client.chat.completions.create(
              messages=self.past_messages[user_id][conversation_id],
              model= model,#"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",#
              max_tokens=5000,
            )
            
            #print(self.past_messages)
            response = chat_completion.choices[0].message.content
            # Update the past messages list with the new chat completion
            print("Response: ", response)
            print("Chat completion", chat_completion)
            response = response.replace("\\_", "_")
            print(response)
            self.add_message(isVisionModel, "assistant", user_text = response, image = image, user_id = user_id, conversation_id=conversation_id)
            
            # Print the assistant's response
            print("Bot: ", response)
            return response
        else:
            return None
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
          model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
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
          model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
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
          model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
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
                    async_nodes[i].trigger()
                    
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
            model = genai.GenerativeModel("gemini-1.5-flash")
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
    