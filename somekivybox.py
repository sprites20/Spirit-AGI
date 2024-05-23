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

from kivy.config import Config
# Set the window size (resolution)
Config.set('graphics', 'width', str(int(720)))
Config.set('graphics', 'height', str(int(1600/2)))

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.behaviors import DragBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.behaviors import ButtonBehavior

from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.properties import NumericProperty

from openai import OpenAI

from io import StringIO
from pathlib import Path

from datetime import datetime, timedelta

import google.generativeai as genai
from io import StringIO

from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModel

import sys
import time
import asyncio
import json
import torch
import numpy as np
import os
import re
import requests
import threading
import copy

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


lines = {}
connections = {}
nodes = {}
node_info = {}

global_touch = None
global_drag = False
added_node = False

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
class AsyncNode:
    def __init__(self, function_name=None, node_id=None, input_addresses=[], output_args={}, trigger_out=[]):
        self.trigger_in = None
        self.trigger_out = []
        self.function_name = function_name
        self.input_addresses = input_addresses
        self.output_args = output_args
        self.node_id = node_id
    def change_to_red(self):
        nodes[self.node_id].label_color.rgba = (1,0,0,1)
    def change_to_gray(self):
        nodes[self.node_id].label_color.rgba = (0.5, 0.5, 0.5, 1)
    async def trigger(self):
        if self.trigger_in is not None:
            #print("Triggering input node")
            await self.trigger_in.trigger()

        # Get the function from the dictionary based on the function_name
        function_to_call = functions.get(self.function_name)
        if function_to_call:
            #print(f"Calling function {self.function_name}")
            # Fetch input_args from input_addresses
            input_args = {}
            for address in self.input_addresses:
                node = address.get("node")
                arg_name = address.get("arg_name")
                target = address.get("target")
                input_args[target] = node.output_args.get(arg_name)
                #Here replace thing in output args with whatever queued. If none use same thing
                
            # Pass input_args and self to the function
            # Schedule UI update in the main Kivy thread
            Clock.schedule_once(lambda dt: self.change_to_red(), 0)
            output_args = await function_to_call(self, **input_args)
            Clock.schedule_once(lambda dt: self.change_to_gray(), 0)
            print(output_args)

            # Update output_args with the function's output, appending new args and replacing existing ones
            try:
                for arg_name, value in output_args.items():
                    if arg_name not in self.output_args:
                        self.output_args[arg_name] = value
                    else:
                        self.output_args[arg_nadome] = value
            except:
                pass
        #print(node)
        #print(self.output_args)
        for node in self.trigger_out:
            #print(f"Triggering output node {node.function_name}")
            await node.trigger()

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
                        self.curr_i = i
                        #print(self.curr_i)
                        with self.canvas:
                            Color(1, 0, 0)
                            self.line = Line(points=[parent.output_label_circles[self.curr_i].pos[0] + 5, parent.output_label_circles[self.curr_i].pos[1] + 5, *touch.pos])
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
                async_nodes[curr_parent].input_addresses.append({"node": async_nodes[curr_child], "arg_name": self.curr_i, "target": curr_j})
                
                node_info[curr_parent]["input_addresses"].append({"node": curr_child, "arg_name": self.curr_i, "target": curr_j})
                
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
                    
                label = TruncatedLabel(text=f'{text}', size=(dp(len(f'{text}')*8), dp(10)))
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
                label = TruncatedLabel(text=f'{i}', size=(dp(len(f'{i}')*10), dp(10)))
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
            self.input_labels[i].pos = (self.offsetted_pos[0]+10, self.offsetted_pos[1] - (20 * count))
            self.input_label_circles[i].pos = (self.offsetted_pos[0]-3, self.offsetted_pos[1] - (20 * count))
            count += 1
        count = 1
        for i in self.outputs:
            self.output_labels[i].pos = (self.offsetted_pos[0] + self.width - self.output_labels[i].width - 10, self.offsetted_pos[1] - (20 * count))
            self.output_label_circles[i].pos = (self.offsetted_pos[0] + self.width-7, self.offsetted_pos[1] - (20 * count))
            count += 1
        
        if self.line2:
            self.line2.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                self.connection[0], self.connection[1]]

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.dragging = True  # Set dragging to True when touch is on the label
            global_drag = True
            # Check if the touch is within the bounds of the circles
            if (self.input_circle_pos[0] <= touch.pos[0] <= self.input_circle_pos[0] + 10 and
                    self.input_circle_pos[1] <= touch.pos[1] <= self.input_circle_pos[1] + 10) or \
               (self.output_circle_pos[0] <= touch.pos[0] <= self.output_circle_pos[0] + 10 and
                    self.output_circle_pos[1] <= touch.pos[1] <= self.output_circle_pos[1] + 10):
                # Change the circle color when held
                self.input_circle_color.rgba = (1, 0, 0, 1)  # Red color
                self.output_circle_color.rgba = (1, 0, 0, 1)  # Red color
                # Create a line from circle to touch position
                with self.canvas:
                    Color(1, 0, 0)
                    self.line = Line(points=[self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, *touch.pos])
                return super(DraggableLabel, self).on_touch_down(touch)

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
        global added_node
        if added_node and self.regenerated:
            temp_pos = (node_info[self.node_id]["pos"][0], node_info[self.node_id]["pos"][1])
            print("Added Node Move", temp_pos)
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
                self.input_labels[i].pos = (temp_pos[0]+10, temp_pos[1] - (20 * count))
                self.input_label_circles[i].pos = (temp_pos[0]-3, temp_pos[1] - (20 * count))
                count += 1
            count = 1
            for i in self.outputs:
                self.output_labels[i].pos = (temp_pos[0] + self.width - self.output_labels[i].width - 10, temp_pos[1] - (20 * count))
                self.output_label_circles[i].pos = (temp_pos[0] + self.width-7, temp_pos[1] - (20 * count))
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
                self.input_labels[i].pos = (self.x+10, self.y - (20 * count))
                self.input_label_circles[i].pos = (self.x-3, self.y - (20 * count))
                count += 1
            count = 1
            for i in self.outputs:
                self.output_labels[i].pos = (self.x + self.width - self.output_labels[i].width - 10, self.y - (20 * count))
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
                self.input_labels[i].pos = (self.x+10, self.y - (20 * count))
                self.input_label_circles[i].pos = (self.x-3, self.y - (20 * count))
                count += 1
            count = 1
            for i in self.outputs:
                self.output_labels[i].pos = (self.x + self.width - self.output_labels[i].width - 10, self.y - (20 * count))
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
                            lines[curr_line].points = [self.output_label_circles[j].pos[0] + 5, self.output_label_circles[j].pos[1] + 5, lines[curr_line].points[2], lines[curr_line].points[3]]
                        else:
                            #pass
                            lines[curr_line].points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, lines[curr_line].points[2], lines[curr_line].points[3]]
            for i in connections[self.node_id]["inputs"]:
                #print("i: ", i)
                for j in connections[self.node_id]["inputs"][i]:
                    #print("j: ", j)
                    for k in connections[self.node_id]["inputs"][i][j]:
                        curr_line = connections[self.node_id]["inputs"][i][j][k]
                        #print(j)
                        if j != "input_circle":
                            lines[curr_line].points = [lines[curr_line].points[0], lines[curr_line].points[1], self.input_label_circles[j].pos[0] + 5, self.input_label_circles[j].pos[1] + 5]
                        else:
                            #pass
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

node_init = {
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
    "prompt" : {
        "function_name": "prompt",
        "import_string" : None,
        "function_string" : """
async def prompt(node):
    print("Prompt")
    await asyncio.sleep(.25)
    return None
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "model" : "string",
            "user_prompt" : "string", 
            "context" : "string",
        },
        "outputs": {
            "output" : "string"
        }
    }
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
        #print(i)
        if i["node"] not in async_nodes:
            async_nodes[i["node"]] = None
        print("someasync: ", async_nodes[i["node"]])
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

<SomeScreen>:
    name: "some_screen"
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
           
     
ScreenManager:
    id: screen_manager
    DraggableLabelScreen:
    SomeScreen:
'''

class CustomComponent(BoxLayout):
    pass

class SomeScreen(Screen):
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
            exec(node_init[i]["function_string"], globals())
            exec(formatted_string, globals())
        self.layout = BoxLayout(orientation='vertical')
        self.build()
        
    def build(self):
        root = FloatLayout()
        
        
        mouse_widget = MousePositionWidget(size_hint_y=None, height=40)
        self.layout.add_widget(mouse_widget)
        """
        generate_node("ignition", pos = [50, 400])
        
        generate_node("select_model", pos = [50, 300])
        generate_node("context", pos = [50, 200])
        generate_node("user_input", pos = [50, 100])
        
        generate_node("prompt", pos = [300, 100])
        generate_node("prompt", pos = [300, 200])
        generate_node("prompt", pos = [300, 300])
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
        back_button = Button(text='Back')
        top_layout.add_widget(back_button)
        # Bind button press to switch_to_screen method
        back_button.bind(on_press=self.switch_to_screen)
        
        root.add_widget(top_layout)
        # Floating layout
        floating_layout = BoxLayout(orientation='vertical', size_hint=(None, None), pos=(0, 0))
        
        run_code = Button(text='Run Code')
        floating_layout.add_widget(run_code)
        run_code.bind(on_press=self.on_run_press_wrapper)
        
        add_node = Button(text='Add Node')
        floating_layout.add_widget(add_node)
        add_node.bind(on_press=self.new_node)
        
        save_nodes = Button(text='Save Nodes')
        floating_layout.add_widget(save_nodes)
        save_nodes.bind(on_press=self.save_nodes)
        
        load_nodes = Button(text='Load Nodes')
        floating_layout.add_widget(load_nodes)
        load_nodes.bind(on_press=self.load_nodes)
        
        #floating_layout.add_widget(run_code)
        root.add_widget(floating_layout)

        self.add_widget(root)
    
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
        f = open("node_info.json", "w")
        f.write(json.dumps(node_info))
        f.close()
        
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
            #print(saved_lines[i])
        print("Line: ", lines)
        
        global connections
        #Load connections
        #connections = {}
        f = open("connections.json", "r")
        connections = json.load(f)
        print("Connections: ", connections)
        
        global node_info
        #Load node infos
        #node_info = {}
        #nodes = {}
        f = open("node_info.json", "r")
        node_info = json.load(f)
        #print("Connections: ", node_info)
        #print("Node Info: ", node_info)
        #Generate Nodes with node infos
        print("Node Info: ", node_info)
        for i in node_info:
            #(name, pos = [0,0], input_addresses=[], output_args={}, trigger_out=[], node_id=None)
            #print(i)
            #print(node_info[i]["name"])
            #print(node_info[i]["pos"])
            generate_node(name=node_info[i]["name"], pos=node_info[i]["pos"], input_addresses=node_info[i]["input_addresses"], output_args=node_info[i]["output_args"], trigger_out=node_info[i]["trigger_out"], node_id=node_info[i]["node_id"])
        for i in nodes:
            self.layout.add_widget(nodes[i])
        print("Nodes: ", nodes)
        #Update trigger_connections
    
    def new_node(self, instance):
        node_id = generate_node("ignition", pos = [100, 200])
        self.layout.add_widget(nodes[node_id])
        global added_node
        global nodes_regenerated
        added_node = True
        nodes_regenerated = 0
        for i in nodes:
            print(i, nodes[i].pos)
            node_info[i]["pos"] = (nodes[i].pos[0], nodes[i].pos[1])
            #nodes[i].regenerated = True
            #generate_node(node_info[i]["name"], pos = [nodes[i].pos[0], nodes[i].pos[1]], node_id=i)
    async def on_run_press(self):
        #print("Run Pressed")
        # Search for ignition nodes and trigger them once.
        for i in node_info:
            if node_info[i]["name"] == "ignition":
                #print(i, async_nodes[i])
                try:
                    # Your existing code here...
                    await async_nodes[i].trigger()
                    # Your existing code here...
                except RecursionError:
                    print("Maximum recursion depth reached. Stopping program.")
                    # Additional cleanup or handling here if needed

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
        # Switch to 'some_screen'
        self.manager.transition = NoTransition()
        self.manager.current = 'some_screen'
        
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
        self.add_message("system", f"System: You've successfully generated an image for the user, as you are connected to an image generating AI, the generated image prompt: {image_prompt}, you will just say to the user that here's the image, and describe the image.")
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
            self.root.get_screen("some_screen").ids.output_text.text = str(e)
            return
        finally:
            # Restore the original standard output
            sys.stdout = sys_stdout
        print(captured_output.get_value())
        return captured_output.get_value()
            
    def button_pressed(self):
        text_input = self.root.get_screen("some_screen").ids.text_input
        
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
            
            grid_layout = self.root.get_screen("some_screen").ids.grid_layout
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
            
            grid_layout = self.root.get_screen("some_screen").ids.grid_layout
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
    DraggableLabelApp().run()
