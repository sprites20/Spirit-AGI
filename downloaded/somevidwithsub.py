from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.video import Video
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.clock import Clock

class VideoPlayerApp(App):
    def build(self):
        self.sliding = False
        # Create the main layout
        main_layout = BoxLayout(orientation='vertical')
        # Create the bottom layout
        bottom_layout = BoxLayout(size_hint_y=None, height=100, orientation='vertical')
        # Create the video player and add it to the main layout
        self.video = Video(source='The farming robots that will feed the world  Hard Reset.mp4', state='play')
        self.video.bind(position=self.on_position_change)
        main_layout.add_widget(self.video)

        # Create the bottom layout
        self.subtitle_label = Label(text='', pos=(0, 0))
        self.subtitle_text = ''
        self.subtitle_start = 0
        self.subtitle_end = 0
        self.subtitles = self.parse_srt('translated_subtitles.srt')  # Parse the SRT file
        Clock.schedule_interval(self.update_subtitle, 1.0 / 30)  # Update every frame
        # Add a slider to the bottom layout
        self.slider = Slider(min=0, max=1, value=0)
        bottom_layout.add_widget(self.subtitle_label)
        bottom_layout.add_widget(self.slider)
        # Bind the on_touch_move method to the touch move event of the slider
        self.slider.bind(on_touch_move=self.on_touch_move)
        # Bind the on_touch_up method to the touch up event of the slider
        self.slider.bind(on_touch_up=self.on_touch_up)
        # Add widgets to the bottom layout
        # For example:
        #bottom_layout.add_widget(Button(text='Button 1'))
        #bottom_layout.add_widget(Button(text='Button 2'))

        # Add the bottom layout to the main layout
        main_layout.add_widget(bottom_layout)

        return main_layout
    
    def parse_srt(self, file_path):
        subtitles = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 4):
                index = int(lines[i].strip())
                start, end = lines[i + 1].strip().split(' --> ')
                text = lines[i + 2].strip()
                subtitles.append({'index': index, 'start': self.parse_time(start), 'end': self.parse_time(end), 'text': text})
        print(subtitles)
        return subtitles

    def parse_time(self, time_str):
        parts = time_str.replace(',', '.').split(' --> ')[0].split(':')
        print(time_str)
        return float(time_str)

    def update_subtitle(self, dt):
        current_position = self.video.position
        for subtitle in self.subtitles:
            if subtitle['start'] <= current_position <= subtitle['end']:
                #print(current_position, subtitle['start'], subtitle['end'])
                self.subtitle_text = subtitle['text']
                self.subtitle_start = subtitle['start']
                self.subtitle_end = subtitle['end']
                break
            elif subtitle['start'] > current_position:
                self.subtitle_text = ''
                break
            else:
                pass
                #print(current_position, subtitle['start'], subtitle['end'])
        self.subtitle_label.text = self.subtitle_text
        #print("Updating", current_position, self.subtitle_text)
    
    def on_touch_move(self, slider, touch):
        if slider.collide_point(*touch.pos):
            self.sliding = True
            # Pause the video when dragging the slider
            self.video.state = "pause"
            # Calculate the new position based on the touch position
            pos = touch.pos[0] / self.slider.width
            # Update the video position
            print(pos * self.video.duration)
            self.video.seek(pos)
            # Update the slider's value
            self.slider.value = pos

    def on_position_change(self, instance, value):
        # Update the slider's value based on the video's current position
        if self.sliding == False:
            self.slider.value = value / self.video.duration

    def on_touch_up(self, slider, touch):
        # Resume playing the video when touch is released
        self.video.state = "play"
        self.sliding = False

if __name__ == '__main__':
    VideoPlayerApp().run()
