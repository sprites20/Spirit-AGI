from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import MDScreen
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import NoTransition
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import os
import datetime

import argparse
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SpkModel
from pydub.silence import split_on_silence
from threading import Thread
import numpy as np

from kivy.clock import Clock  
from kivymd.uix.label import MDLabel
from kivy.core.audio import SoundLoader
import pyttsx3
import json
#from android.permissions import request_permissions, Permission
#from kivy.core.window import Window

#Window.size = (360, 640)  # Adjust the size based on your preference

from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment

import numpy as np
#import wave

import cohere 
co = cohere.Client('MLZXavfC2EpNaW3dYRG5KwWPcMIvBUyabF1DPBgw') # This is your trial API key
tts = pyttsx3.init()
q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def save_audio(filename, audio_data, samplerate):
    audio_segment = AudioSegment(
        data=audio_data, sample_width=2, frame_rate=samplerate, channels=1
    )
    audio_segment.export(filename, format="wav")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l", "--list-devices", action="store_true",
    help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    sys.exit(0)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    "-f", "--filename", type=str, metavar="FILENAME",
    help="audio file to store recording to")
parser.add_argument(
    "-d", "--device", type=int_or_str,
    help="input device (numeric ID or substring)")
parser.add_argument(
    "-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument(
    "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
args = parser.parse_args(remaining)



# Function to modify already saved audio file
def modify_saved_audio(input_filename, output_filename, samplerate):
    audio_segment = AudioSegment.from_wav(input_filename)

    # Split the audio based on silence
    # Adjust silence detection parameters as needed
    segments = split_on_silence(audio_segment, silence_thresh=-50, keep_silence=100)

    # Concatenate non-silent segments
    concatenated_audio = AudioSegment.silent()
    for i, segment in enumerate(segments):
        concatenated_audio += segment

    # Save the concatenated audio
    concatenated_audio.export(output_filename, format="wav")



KV = '''
<DrawerClickableItem@MDNavigationDrawerItem>
    focus_color: "#e7e4c0"
    text_color: "#4a4939"
    icon_color: "#4a4939"
    ripple_color: "#c5bdd2"
    selected_color: "#0c6c4d"

<DrawerLabelItem@MDNavigationDrawerItem>
    text_color: "#4a4939"
    icon_color: "#4a4939"
    focus_behavior: False
    selected_color: "#4a4939"
    _no_ripple_effect: True

<MyClickableButton>:
    size_hint_y: None
    height: "40dp"

<CustomLabel>:
    size_hint_y: None
    height: self.texture_size[1] + 10
    text_size: self.width, None
    padding: '5dp'
    canvas.before:
        Color:
            rgba: 0.7, 0.7, 0.7, 1  # Background color
        Rectangle:
            pos: self.pos
            size: self.size
    multiline: True
    
MDScreen:
    MDNavigationLayout:
        swipe_distance: self.width / 2
        swipe_edge_width: 0
        MDScreenManager:
            id: screen_manager
            swipe: False  # Disable swipe transitions
            MDScreen:
                name: 'main_screen'
                transition: "NoTransition"  # Disable transition
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: (1, 1)

                    BoxLayout:
                        orientation: 'vertical'
                        size_hint: (1, 1)  # 1/3 of the width
                        padding: '10dp'
                        spacing: '10dp'
                        pos_hint: {"top": 0.9}  # Adjust the top position here
                        MDRaisedButton:
                            text: 'Chatbot'
                            on_release: app.run_vosk_stt_thread()

                        MDRaisedButton:
                            text: 'File Manager'
                            on_release: app.switch_screen('screen2')
                        ScrollView:
                            GridLayout:
                                cols: 1
                                spacing: '10dp'
                                padding: '10dp'
                                size_hint_y: None
                                height: self.minimum_height  # Make the height dynamic
                                MDRaisedButton:
                                    text: 'File Retrieval Agent'
                                    size_hint: None, None
                                    width: self.width
                                    height: '48dp'
                                    anchor_x: 'center'
                                    anchor_y: 'center'
                                MDRaisedButton:
                                    text: 'Facebook Scraper'
                                    size_hint: None, None
                                    width: self.width
                                    height: '48dp'
                                    anchor_x: 'center'
                                    anchor_y: 'center'
                                MDRaisedButton:
                                    text: 'Music Player'
                                    size_hint: None, None
                                    width: self.width
                                    height: '48dp'
                                    anchor_x: 'center'
                                    anchor_y: 'center'

                    MDScreen:
                        name: 'screen1'
                        transition: "NoTransition"  # Disable transition
                        FloatLayout:
                            size_hint: (1, 1)  # 2/3 of the width
                            pos_hint: {"top": 1, "right": 1}  # Top right corner
                            ScrollView:
                                pos_hint: {"top": 0.9}  # Adjust the top position here
                                GridLayout:
                                    cols: 1
                                    spacing: '10dp'
                                    padding: '10dp'
                                    size_hint_y: None
                                    id: messages
                                    height: self.minimum_height  # Make the height dynamic
                                    MDRaisedButton:
                                        text: 'Chatbot Message 1'
                                        size_hint: None, None
                                        width: '200dp'
                                        height: '48dp'
                                        pos_hint: {"left": 1}  # Align to the left
                                    MDRaisedButton:
                                        text: 'User Message 1'
                                        size_hint: None, None
                                        width: '200dp'
                                        height: '48dp'
                                        pos_hint: {"right": 1}  # Align to the right
                                    # Add more messages here
                            BoxLayout:
                                orientation: 'horizontal'
                                size_hint_y: None
                                height: '48dp'
                                MDRaisedButton:
                                    text: "Send"
                                    on_release: app.send_message()
                                TextInput:
                                    id: message_input
                                    hint_text: "Type your message here"
            MDScreen:
                name: 'screen1'
                transition: "NoTransition"  # Disable transition
                BoxLayout:
                    orientation: "vertical"
                    size_hint: (1, 0.9)  # 2/3 of the width
                    pos_hint: {"top": 1, "right": 1}  # Top right corner
                    ScrollView:
                        pos_hint: {"top": 0.9}  # Adjust the top position here
                        GridLayout:
                            cols: 1
                            spacing: '10dp'
                            padding: '10dp'
                            size_hint_y: None
                            height: self.minimum_height  # Make the height dynamic
                            MDRaisedButton:
                                text: 'Commander of Agents'
                                size_hint: None, None
                                width: '200dp'
                                height: '48dp'
                                pos_hint: {"left": 1}  # Align to the left
                            
                            # Add more messages here
                    BoxLayout:
                        id: button_layout
                        orientation: 'horizontal'
                        size_hint_y: None
                        height: '48dp'
                        MDRaisedButton:
                            text: "Send"
                            on_release: app.send_message()
                        TextInput:
                            id: message_input
                            hint_text: "Type your message here"
            MDScreen:
                name: 'screen2'
                transition: "NoTransition"  # Disable transition
                BoxLayout:
                    orientation: "vertical"
                    spacing: "8dp"
                    pos_hint: {"top": 0.9}  # Adjust the top position here
                    BoxLayout:
                        size_hint_y: None
                        height: "40dp"
                        spacing: "8dp"

                        Button:
                            text: "Up"
                            on_release: app.navigate_up()

                        Label:
                            id: current_directory_label  # Add an id to reference in Python

                    ScrollView:
                        GridLayout:
                            id: file_list
                            cols: 4
                            spacing: "8dp"
                            size_hint_y: None
                            height: self.minimum_height
                            padding: dp(16)
                            pos_hint: {"top": 1}  # Adjusted pos_hint
        MDTopAppBar:
            title: "Talk to your Agents!"
            elevation: 4
            pos_hint: {"top": 1}
            md_bg_color: "#e7e4c0"
            specific_text_color: "#4a4939"
            left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]
        MDNavigationDrawer:
            id: nav_drawer
            radius: (0, 16, 16, 0)
            MDNavigationDrawerMenu:
                MDNavigationDrawerHeader:
                    title: "Header title"
                    title_color: "#4a4939"
                    text: "Header text"
                    spacing: "4dp"
                    padding: "12dp", 0, 0, "56dp"
                MDNavigationDrawerLabel:
                    text: "Mail"
                DrawerClickableItem:
                    icon: "gmail"
                    right_text: "+99"
                    text_right_color: "#4a4939"
                    text: "Inbox"
                DrawerClickableItem:
                    icon: "send"
                    text: "Outbox"
                MDNavigationDrawerDivider:
                MDNavigationDrawerLabel:
                    text: "Labels"
                DrawerLabelItem:
                    icon: "information-outline"
                    text: "Label"
                DrawerLabelItem:
                    icon: "information-outline"
                    text: "Label"
'''
class CustomLabel(Label):
    pass

class MyClickableButton(Button):
    max_text_length = 20  # Define max_text_length at the class level
    file_name = ""
    size_text = ""
    date_modified_text = ""
    type_text = ""

    spacing_height = 8  # Adjust the spacing value as needed

    def __init__(self, is_header=False, is_visible=True, **kwargs):
        super(MyClickableButton, self).__init__(**kwargs)
        self.bind(on_release=self.on_button_click)

        # Check if the button is a header (group label)
        self.is_header = is_header

        if not self.is_header:
            # Create a GridLayout inside the button for regular buttons
            layout = GridLayout(cols=4, spacing=(0, self.spacing_height))  # Adjust cols based on your needs

            # Add labels to the GridLayout
            layout.add_widget(Label(text=self.file_name, size_hint_x=0.6))
            layout.add_widget(Label(text=self.size_text, size_hint_x=0.2))
            layout.add_widget(Label(text=self.date_modified_text, size_hint_x=0.2))
            layout.add_widget(Label(text=self.type_text, size_hint_x=0.2))

            # Add the GridLayout to the button
            self.add_widget(layout)

        self.opacity = 1 if is_visible else 0

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def on_button_click(self, instance):
        # Handle button click here
        pass
class Example(MDApp):
    path_history = []  # Define path_history as a class attribute
    mode = 0
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.primary_hue = "800"
        self.theme_cls.accent_palette = "Amber"
        self.theme_cls.accent_hue = "500"
        
        return Builder.load_string(KV)
    
    def local_retrieve(self, query):
        # Assume you have a query and its vector representation
        query = "Plot normal distribution"
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

        # Calculate cosine similarity between the query and each sentence
        similarities = []
        for idx, data in output_json.items():
            sentence_vectors = np.array(data["vectors"])
            similarity = cosine_similarity(query_embedding, sentence_vectors)[0][0]
            similarities.append((idx, similarity))

        # Sort the sentences based on similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Print the top N most relevant sentences
        top_n = 3
        print(f"\nTop {top_n} most relevant sentences to the query '{query}':")
        for idx, similarity in similarities[:top_n]:
            print(f"File number: {output_json[idx]['file_number']}")
            print(f"File path: {mapping_json[output_json[idx]['file_number']]}")
            print(f"Sentence: {output_json[idx]['sentences']}")
            print(f"Similarity: {similarity}\n")

        
    def on_start(self):
        self.mode = 0
        self.path_history = [os.getcwd()]
        self.load_files()
        self.update_current_directory_label()  # Update the label text here
    
    def cosine_similarity_average(self, speaker_embeddings, target_speaker):
        lowest_similarity = {}

        for speaker, embeddings in speaker_embeddings.items():
            # Get the lowest similarity among the two embeddings for each speaker
            similarities = [self.cosine_dist(target_speaker, embedding) for embedding in embeddings]
            lowest_similarity[speaker] = min(similarities)

        return lowest_similarity
    
    def recog_speaker(self, target_speaker):
        speakers = {
            "Ian": [[-0.645543, 1.267236, 1.739462, -0.717491, -0.157087, 0.147635, -1.308505, -0.446466, 0.116764, -0.115046, 0.376392, 0.62511, 0.554749, 0.871882, 1.705446, 1.346732, -0.237086, 0.554086, 0.171249, 0.035732, 0.079214, -0.577399, 1.605019, -0.872605, -0.80465, -0.402827, -0.621014, -0.13613, 1.766777, 1.253641, -1.048572, -1.723634, -0.525028, -0.512419, 0.979154, -0.29935, -1.11108, 1.460288, -0.492389, -0.165662, -0.274988, 0.458642, 1.453099, 1.092062, -0.856726, 0.724769, 0.423962, -0.774903, -0.434743, -0.083244, 0.685712, -0.579763, -0.160493, 0.699621, -0.95782, -1.056444, -0.218858, 0.508616, -0.441598, 0.140081, 0.870923, -1.356405, -0.179892, -0.495612, -0.165929, 0.162548, -0.490384, 0.044856, -0.585081, 2.214094, 0.511557, -2.132176, -0.329827, 1.419002, -1.156591, -0.265651, -1.553596, -0.50643, 0.627002, -1.194909, -0.253832, 0.115579, 0.164481, -0.543525, -0.657609, 0.529603, 0.917261, 1.276905, 2.072457, 0.501246, -0.229274, 0.554694, -1.703213, -0.693821, 0.768317, -0.404479, 2.06889, -1.26462, -0.019318, 0.715243, 1.138082, -1.728924, -0.714421, -1.267921, 1.681902, -1.716266, -0.074632, -2.936986, -2.350122, 0.001327, -0.382891, -0.688902, 1.322296, -0.987495, 1.975746, -0.44887, 0.185008, 0.067595, 0.665363, 0.246385, 0.719629, 0.506032, -0.988654, 0.606328, -1.949532, 1.727559, -1.032074, -0.772542],
                    [-0.683516, 0.722179, 1.651159, -0.311776, -0.35272, -0.542711, -0.169784, 0.146419, 0.639174, 0.260786, 0.512685, -0.567375, 0.510885, 1.081993, 0.730045, 1.644301, -0.388575, 0.594761, 0.580934, 1.701163, 0.542753, -0.030902, 0.940672, -0.681181, -0.961269, -0.953732, 0.342842, 0.212761, 1.010038, 0.789226, -0.440633, -1.639356, 0.098124, -0.453873, -0.1269, -0.831008, -1.336311, 1.838328, -1.500506, 0.398561, -0.139225, 0.602066, 1.217693, -0.28669, -1.240536, 0.828214, -0.385781, -1.585939, -0.253948, 0.6254, -1.144157, -1.09649, -1.247936, -0.164992, -1.131125, -0.827816, 1.595752, 1.22196, -0.260766, -0.053225, 0.372862, -0.496685, 0.559101, 0.313831, 0.906749, -0.911119, -0.718342, 0.731359, -0.060828, 0.889468, 0.870002, -1.046849, 0.358473, 1.403957, -0.55995, 0.544278, 0.252579, 0.176449, -0.973618, -1.316356, -1.39273, -0.397281, -1.244906, -2.552846, -0.056479, 0.00252, -0.071661, 0.549343, -0.563582, 0.298601, -1.599536, 0.060805, -1.131684, -0.236406, 0.10192, -0.05143, 2.822287, 0.298605, 0.027687, 1.805171, 0.535367, -0.750344, 0.195215, -2.74342, -0.240448, -1.853602, 0.667115, -1.152912, -1.458451, -0.463823, -1.081316, 1.07476, 1.69582, 0.083853, 0.208222, -0.203687, -0.761975, 2.021879, 2.07578, 0.214109, 1.010975, -0.535104, -1.102454, 1.422523, -1.389488, 2.282245, 0.526214, -0.289677],
                    [-0.645543, 1.267236, 1.739462, -0.717491, -0.157087, 0.147635, -1.308505, -0.446466, 0.116764, -0.115046, 0.376392, 0.62511, 0.554749, 0.871882, 1.705446, 1.346732, -0.237086, 0.554086, 0.171249, 0.035732, 0.079214, -0.577399, 1.605019, -0.872605, -0.80465, -0.402827, -0.621014, -0.13613, 1.766777, 1.253641, -1.048572, -1.723634, -0.525028, -0.512419, 0.979154, -0.29935, -1.11108, 1.460288, -0.492389, -0.165662, -0.274988, 0.458642, 1.453099, 1.092062, -0.856726, 0.724769, 0.423962, -0.774903, -0.434743, -0.083244, 0.685712, -0.579763, -0.160493, 0.699621, -0.95782, -1.056444, -0.218858, 0.508616, -0.441598, 0.140081, 0.870923, -1.356405, -0.179892, -0.495612, -0.165929, 0.162548, -0.490384, 0.044856, -0.585081, 2.214094, 0.511557, -2.132176, -0.329827, 1.419002, -1.156591, -0.265651, -1.553596, -0.50643, 0.627002, -1.194909, -0.253832, 0.115579, 0.164481, -0.543525, -0.657609, 0.529603, 0.917261, 1.276905, 2.072457, 0.501246, -0.229274, 0.554694, -1.703213, -0.693821, 0.768317, -0.404479, 2.06889, -1.26462, -0.019318, 0.715243, 1.138082, -1.728924, -0.714421, -1.267921, 1.681902, -1.716266, -0.074632, -2.936986, -2.350122, 0.001327, -0.382891, -0.688902, 1.322296, -0.987495, 1.975746, -0.44887, 0.185008, 0.067595, 0.665363, 0.246385, 0.719629, 0.506032, -0.988654, 0.606328, -1.949532, 1.727559, -1.032074, -0.772542],
                    [-0.728723, 0.231748, 1.115237, -0.333376, -1.222962, -0.514356, 0.184838, 0.593837, -0.205287, 2.280802, -0.492246, 0.748175, -0.440483, 0.519464, 0.253391, 1.724434, -0.114792, 1.038521, 0.125444, 0.078519, -0.002113, 0.517065, 1.867799, 2.171547, 0.674426, -0.082566, -0.898308, -1.564627, 1.447941, 0.36099, -1.685272, -0.489678, 0.773777, -1.113652, -0.242769, -0.019092, 1.101615, 1.144882, 1.290024, 0.527264, -0.372259, 1.420498, 0.228505, -0.250007, 0.511168, -0.613636, -0.166762, -1.648969, -1.319836, -0.035552, -1.364019, -1.345776, -0.419131, 0.319625, -0.582904, 0.232689, 0.452173, 0.563885, 1.80914, -1.187325, -0.097138, 1.094703, -1.220001, -0.706769, -2.780749, -0.110709, -0.478987, 0.152605, -0.766997, -0.581186, 1.173153, 1.762713, 2.374964, 1.615848, 1.226277, 1.355428, -0.765733, -0.198867, -1.143859, -2.059099, 0.516039, 0.033704, -1.46795, -0.051112, -0.342946, 0.867685, -0.184492, -1.024215, -1.041677, -1.288777, -0.198457, -1.040588, -1.182497, -1.129668, 0.238964, 0.100704, 2.054156, 0.051816, 0.165922, -0.31556, -1.311156, 0.928618, 0.897451, -1.467812, 0.234959, -1.163902, 0.08363, 0.581422, -0.998858, 0.178401, 0.120183, 0.485549, 1.438759, -0.712948, 1.027102, -0.11764, -0.595998, 0.226724, -1.435966, -0.241586, -0.637766, 0.433057, -0.90344, 0.78472, -0.161674, 0.662773, 0.991656, -2.124147],
                    [-0.586079, -0.095003, 1.111695, -1.029107, -0.011695, 0.710173, -1.16822, -1.076212, 0.099983, 2.018286, 0.883726, -0.435856, -0.576487, 0.337998, 0.81329, 0.923432, 0.5851, 1.106504, -0.874023, 0.551154, 0.348427, 0.422592, 0.85514, -0.311744, 0.263824, -0.277768, -0.762957, 0.64646, 0.31419, 0.922777, -1.28617, -0.095353, -0.142192, -0.403755, -0.030829, -1.602873, 1.391456, 1.149409, 0.153811, -0.22231, 1.131615, 1.247838, -0.648151, 0.050896, -1.590242, 0.899025, 0.342044, -1.25762, 0.769508, 1.64223, -0.415507, -1.195839, -1.724025, -0.014763, -0.918404, -0.03256, 0.449659, -0.365694, -2.411637, -1.659071, -0.274899, -0.243476, -1.925734, -1.187865, -0.552587, -1.584616, -0.304403, 1.0202, 1.207309, 0.45385, 1.319396, -0.502503, -0.281457, -0.313889, -0.57542, 0.142528, -0.051609, 0.90746, -0.53427, -0.029579, 1.916388, -0.137778, -1.621211, 0.220487, -0.247424, -0.275082, -1.327777, 0.64919, -0.109337, -0.666244, -1.617283, 1.338911, -1.195554, 1.680849, 0.759325, -0.100275, 0.414661, -1.758352, -0.044392, 2.355603, -1.551207, 1.056203, 0.976233, 0.548299, 0.985702, -1.52446, -0.504073, -1.626964, -0.999765, -1.179184, 0.787754, 1.648664, -0.574964, -2.663006, 1.870389, 0.884753, 0.486392, 0.155071, -0.605704, -0.176426, -0.326237, -0.774213, 0.738778, 1.014247, 1.458341, 0.50901, 0.217568, 0.552388],
            ],
        }
        #Iterate in speakers and get average cosine similarity and return average cosine similarity for each and the lowest
        lowest_similarity = self.cosine_similarity_average(speakers, target_speaker)

        # Print the results
        print("Lowest Similarities:")
        dspeaker, thesim = None, None
        for speaker, lowest_sim in lowest_similarity.items():
            print(f"{speaker}: {lowest_sim}")
            dspeaker, thesim = speaker, lowest_sim
            break
        dactual = "Unknown User"
        #print(cosine_similarity_average(speakers, target_speaker))
        if thesim < 0.60:
            dactual = dspeaker
        return dactual
        
    def cosine_dist(self, x, y):
        nx = np.array(x)
        ny = np.array(y)
        return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)
         
    def run_vosk_stt_thread(self):
        # Create a thread for vosk_stt and start it
        vosk_thread = Thread(target=self.vosk_stt)
        vosk_thread.start()
    
    def change_text_input(self, result, user):
        # Schedule the UI update using Kivy's Clock class
        Clock.schedule_once(lambda dt: self.update_text_input(result, user))
    
    def update_text_input(self, result, user):
        screen_manager = self.root.ids.screen_manager
        screen1 = screen_manager.get_screen('main_screen').children[0].children[0].children[0]  # Get the 'screen1' widget
        message_input = screen1.children[0].children[0]

        if self.mode == 1:
            if result == "yes":
                self.mode = 1
                self.send_message(user)
                message_input.text = ""
                self.mode = 0
            elif result != "":
                #self.mode = 0
                message_input.text += " " + result
                self.play_offline_tts("Is that all?")
                self.mode = 1
        elif result != "":
            #self.mode = 0
            message_input.text += " " + result
            self.play_offline_tts("Is that all?")
            self.mode = 1
        print(self.mode)
    
    def new_memory(self, speaker_vector=None, speaker=None, text=None, image_bytes=None, image_format=None, image_vector=None, source=None, place=None, message_type=None):
        current_time = datetime.datetime.now()
        datetime_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        epoch_time = int(current_time.timestamp())

        json_data = {
            "speaker_vector": speaker_vector,
            "speaker": speaker,
            "text": text,
            "image_in_bytes": image_bytes,
            "image_format": image_format,
            "image_vector": image_vector,
            "datetime": datetime_str,
            "epoch_time": epoch_time,
            "source": source,
            "place": place,
            "type": message_type
        }

        self.save_json_to_file(json_data, "memories/", speaker + "_" + datetime_str)

    def save_json_to_file(self, json_data, folder_path, file_name):
        file_path = f"{folder_path}/{file_name}.json"
        with open(file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"JSON saved to {file_path}")
    
        
    def vosk_stt(self):
        try:
            if args.samplerate is None:
                device_info = sd.query_devices(args.device, "input")
                args.samplerate = int(device_info["default_samplerate"])

            model = Model(f"vosk-model-small-en-us-0.15")
            SPK_MODEL_PATH = "vosk-model-spk-0.4"
            
            spk_model = SpkModel(SPK_MODEL_PATH)
            spk_sig = [-0.645543, 1.267236, 1.739462, -0.717491, -0.157087, 0.147635, -1.308505, -0.446466, 0.116764, -0.115046, 0.376392, 0.62511, 0.554749, 0.871882, 1.705446, 1.346732, -0.237086, 0.554086, 0.171249, 0.035732, 0.079214, -0.577399, 1.605019, -0.872605, -0.80465, -0.402827, -0.621014, -0.13613, 1.766777, 1.253641, -1.048572, -1.723634, -0.525028, -0.512419, 0.979154, -0.29935, -1.11108, 1.460288, -0.492389, -0.165662, -0.274988, 0.458642, 1.453099, 1.092062, -0.856726, 0.724769, 0.423962, -0.774903, -0.434743, -0.083244, 0.685712, -0.579763, -0.160493, 0.699621, -0.95782, -1.056444, -0.218858, 0.508616, -0.441598, 0.140081, 0.870923, -1.356405, -0.179892, -0.495612, -0.165929, 0.162548, -0.490384, 0.044856, -0.585081, 2.214094, 0.511557, -2.132176, -0.329827, 1.419002, -1.156591, -0.265651, -1.553596, -0.50643, 0.627002, -1.194909, -0.253832, 0.115579, 0.164481, -0.543525, -0.657609, 0.529603, 0.917261, 1.276905, 2.072457, 0.501246, -0.229274, 0.554694, -1.703213, -0.693821, 0.768317, -0.404479, 2.06889, -1.26462, -0.019318, 0.715243, 1.138082, -1.728924, -0.714421, -1.267921, 1.681902, -1.716266, -0.074632, -2.936986, -2.350122, 0.001327, -0.382891, -0.688902, 1.322296, -0.987495, 1.975746, -0.44887, 0.185008, 0.067595, 0.665363, 0.246385, 0.719629, 0.506032, -0.988654, 0.606328, -1.949532, 1.727559, -1.032074, -0.772542]
       
            with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device,
                    dtype="int16", channels=1, callback=callback):
                print("#" * 80)
                print("Press Ctrl+C to stop the recording")
                print("#" * 80)

                rec = KaldiRecognizer(model, args.samplerate)
                rec.SetSpkModel(spk_model)
                recording_started = False
                audio_data = b""

                while True:
                    data = q.get()

                    if rec.PartialResult() != "":
                        if not recording_started:
                            print("Recording started.")
                            recording_started = True

                    if recording_started:
                        audio_data += data

                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        print(res)
                        print("Recording stopped.")
                        result = res["text"]
                        #self.change_text_input(result)
                        print(result)
                        print("Printed results")
                        
                        
                        if "spk" in res:
                            print("X-vector:", res["spk"])
                            print("Speaker distance:", self.cosine_dist(spk_sig, res["spk"]),
                                "based on", res["spk_frames"], "frames")
                            print("Recognizing speaker...")
                            dspeaker = self.recog_speaker(res["spk"])
                            self.new_memory(speaker = dspeaker, text = result, speaker_vector = res["spk"])
                            self.change_text_input(result, dspeaker + ": ")
                            if rec.Result() != "":
                                # Save the entire audio
                                save_audio("output_audio.wav", audio_data, args.samplerate)

                                # After recording is done and you have an output audio file
                                input_audio_filename = "output_audio.wav"
                                output_audio_filename = "modified_output_audio.wav"

                                modify_saved_audio(input_audio_filename, output_audio_filename, args.samplerate)

                                recording_started = False
                                audio_data = b""

        except KeyboardInterrupt:
            print("\nDone")
        except Exception as e:
            print(type(e).__name__ + ": " + str(e))

    
    def load_files(self, path="."):
        file_list_layout = self.root.ids.file_list
        file_list_layout.clear_widgets()

        try:
            files = os.listdir(path)
        except OSError:
            print("Error reading directory")

        # Organize files into groups based on file extension
        file_groups = {}
        for file_name in files:
            file_path = os.path.join(path, file_name)
            file_extension = os.path.splitext(file_name)[1].lower()  # Convert to lowercase for case-insensitivity

            if file_extension not in file_groups:
                file_groups[file_extension] = []

            file_groups[file_extension].append(file_name)

        # Add headers and buttons for each group
        for file_extension, file_names in file_groups.items():
            if file_extension.startswith("."):
                file_extension = file_extension[1:]  # Remove leading dot
            header_label = MyClickableButton(text=f"Group: {file_extension}", is_header=True)
            file_list_layout.add_widget(header_label)
            file_list_layout.add_widget(MyClickableButton(text="", is_visible=False))
            file_list_layout.add_widget(MyClickableButton(text="", is_visible=False))
            file_list_layout.add_widget(MyClickableButton(text="", is_visible=False))
            
            for file_name in file_names:
                file_path = os.path.join(path, file_name)

                # Truncate the file name if too long
                truncated_name = self.truncate_name(file_name, MyClickableButton.max_text_length)

                # Get additional file information
                file_size = os.path.getsize(file_path)
                file_date_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime(
                    "%Y-%m-%d %H:%M:%S")
                file_type = "File" if os.path.isfile(file_path) else "Directory"

                # Create clickable labels for each piece of information
                labels = [
                    MyClickableButton(text=truncated_name),
                    MyClickableButton(text=str(file_size) + " bytes"),
                    MyClickableButton(text=file_date_modified),
                    MyClickableButton(text=file_type)
                ]

                for label in labels:
                    file_list_layout.add_widget(label)
                    # Set on_release callback for each label
                    label.bind(on_release=lambda instance, name=truncated_name, path=path: self.row_clicked(name, path))

        # Manually update the height of the GridLayout
        file_list_layout.height = file_list_layout.minimum_height

        # Update the current directory label
        self.update_current_directory_label()

    def truncate_name(self, name, max_length=20):
        if len(name) <= max_length:
            return name
        else:
            return name[:max_length - 3] + "..."

    def row_clicked(self, file_name, path):
        selected_path = os.path.join(path, file_name)

        if os.path.isdir(selected_path):
            self.path_history.append(selected_path)
            self.load_files(selected_path)

    def navigate(self, direction):
        current_index = self.path_history.index(os.getcwd())
        new_index = current_index + direction

        if 0 <= new_index < len(self.path_history):
            new_path = self.path_history[new_index]
            self.load_files(new_path)

    def navigate_up(self):
        if len(self.path_history) > 1:
            parent_path = os.path.dirname(self.path_history[-1])
            self.path_history.append(parent_path)
            self.load_files(parent_path)

    def get_current_directory(self):
        return self.path_history[-1]

    def update_current_directory_label(self):
        # Update the text of the current directory label
        self.root.ids.current_directory_label.text = self.get_current_directory()
        
    def switch_screen(self, screen_name):
        screen_manager = self.root.ids.screen_manager
        current_screen = screen_manager.current

        # Cancel the transition
        screen_manager.transition = NoTransition()

        # Switch to the target screen
        screen_manager.current = screen_name
    
    def play_tts_audio(self, tts_audio):
        # Play the TTS audio
        tts_audio.seek(0)  # Ensure the stream is at the beginning
        audio_segment = AudioSegment.from_mp3(tts_audio)
        audio_segment.export("tts_audio.wav", format="wav")  # Export to a temporary WAV file
        sound = SoundLoader.load("tts_audio.wav")
        if sound:
            sound.play()
    
    def play_offline_tts(self, text):
        # Initialize the TTS engine

        # Set properties (optional)
        tts.setProperty("rate", 150)  # Speech rate (words per minute)
        tts.setProperty("volume", 2.0)  # Volume level (0.0 to 1.0)

        # Convert text to speech and play
        tts.say(text)
        tts.runAndWait()
        
    def send_message(self, user):
        screen_manager = self.root.ids.screen_manager
        print("Printing", screen_manager.get_screen('main_screen').children[0].children[0].children)
        screen1 = screen_manager.get_screen('main_screen').children[0].children[0].children[0] # Get the 'screen1' widget
        print("Printing", screen1)
        # Get a reference to the TextInput widget with id 'message_input'
        message_input = screen1.children[0].children[0]
        print(message_input)
        # Get the input text from the TextInput widget
        input_text = message_input.text
        message_layout = screen1.children[1].children[0]
        
        user_label = CustomLabel(text= user + input_text)
        message_layout.add_widget(user_label)
        print("Sending to cohere: " + user + input_text)
        response = co.chat( 
          message=user + input_text,
          prompt_truncation='auto',
          connectors=[{"id": "web-search"}]
        ) 
        
        some_res = response.text
        print(some_res)
        
        cohere_label = CustomLabel(text="Cohere: " + some_res)
        self.new_memory(speaker = "Cohere", text = some_res)
        message_layout.add_widget(cohere_label)

        try:
            # Convert text to speech using gTTS
            """
            tts = gTTS(text=some_res, lang='en')
            tts_stream = BytesIO()
            tts.write_to_fp(tts_stream)

            # Play the TTS audio
            self.play_tts_audio(tts_stream)
            """
            self.play_offline_tts(some_res)
        except:
            pass
        
Example().run()