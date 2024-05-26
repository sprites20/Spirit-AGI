from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen, ScreenManager

# Custom component with Label on the left and Button on the right
class NewNodeComponent(BoxLayout):
    def __init__(self, text, **kwargs):
        super(NewNodeComponent, self).__init__(orientation='horizontal', **kwargs)
        self.text = text
        self.label = Label(text=text, size_hint_x=0.7, halign='left', valign='middle')
        self.label.bind(size=self.label.setter('text_size'))  # Ensure the text size matches the label size
        self.button = Button(text="Add", size_hint_x=0.3)
        self.button.bind(on_press=self.button_on_press)
        self.add_widget(self.label)
        self.add_widget(self.button)
    def button_on_press(self, instance):
        try:
            print(self.text)
            app = App.get_running_app()
            app.root.get_screen('draggable_label_screen').new_node(node_name=text)
        except Exception as e:
            print(e)

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
    "display_output" : {
        "function_name": "display_output",
        "import_string" : None,
        "function_string" : """
async def display_output(node, user_input, output, instruct_type):
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
        
        user_custom_component = NewNodeComponent(img_source="images/user_logo.png", txt=user_message)
        bot_custom_component = NewNodeComponent(img_source="images/bot_logo.png", txt=bot_message)
        
        grid_layout = app.root.get_screen("some_screen").ids.grid_layout

        # Remove old image component if it exists
        for child in grid_layout.children[:]:
            if isinstance(child, CustomImageComponent):
                grid_layout.remove_widget(child)
        
        grid_layout.add_widget(user_custom_component)
        grid_layout.add_widget(bot_custom_component)
        
        if instruct_type == 1:
            image_component = CustomImageComponent(img_source="images/generated_image.jpeg")
            grid_layout.add_widget(image_component)

    # Schedule the update_ui function to run on the main thread
    Clock.schedule_once(update_ui)
        """,
        "description" : None,
        "documentation" : None,
        "inputs" : {
            "user_input" : "string",
            "output" : "string",
            "instruct_type" : "num",
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
async def prompt(node, model=None, user_prompt=None, context=None):
    app = MDApp.get_running_app()
    print("Prompt")
    print(model, user_prompt, context)
    await asyncio.sleep(.25)
    user_text = user_prompt
    instruct_type = app.get_instruct_type(user_text)
    if instruct_type == 1:
        app.generate_image_prompt(user_text)
    # Continue the conversation            
    response = app.continue_conversation()
    print("output: ", response)
    return {"output" : response, "instruct_type" : instruct_type}
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
        }
    }
}

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
        main_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        main_layout.bind(minimum_height=main_layout.setter('height'))
        
        # Add custom components to the main layout
        for i in node_init:  # Adding multiple custom components
            custom_component = NewNodeComponent(text=f"{i}")
            custom_component.size_hint_y = None
            custom_component.height = 50
            main_layout.add_widget(custom_component)
        
        # Add the main layout to the scroll view
        main_scroll.add_widget(main_layout)
        
        # Add the back button and scroll view to the screen layout
        screen_layout.add_widget(back_box)
        screen_layout.add_widget(main_scroll)
        
        # Add the screen layout to the screen
        self.add_widget(screen_layout)
    def back_button_on_press(self, instance):
        app.root.current = "draggable_label_screen"
class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(SelectNodeScreen(name='node_screen'))
        return sm

if __name__ == '__main__':
    MyApp().run()
