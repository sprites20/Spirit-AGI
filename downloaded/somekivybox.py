"""
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

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.behaviors import DragBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.widget import Widget

# import the time module
import time

lines = {}
connections = {}
nodes = {}
node_info = {}
global_touch = None

def is_point_in_ellipse(point, center, size):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    a = size[0]
    b = size[1]
    return (dx*dx) / (a*a) + (dy*dy) / (b*b) <= 1

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
            print("Touched TouchableRectangle")
            #print("Touched DraggableLabel:", self.name)

            if not hasattr(touch, 'touchdown'):
                touch.touchdown = True
                print(parent.name)
                for i in parent.output_label_circles:
                    print(i, parent.output_label_circles[i].pos, touch.pos)
                    if (parent.output_label_circles[i].pos[0] <= touch.pos[0] <= parent.output_label_circles[i].pos[0] + 10 and
                        parent.output_label_circles[i].pos[1] <= touch.pos[1] <= parent.output_label_circles[i].pos[1] + 10):
                        # Change the circle color when held
                        print(True)
                        self.curr_i = i
                        print(self.curr_i)
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
            
            for child in self.parent.parent.children:
                if isinstance(child, DraggableLabel) and child != self:
                    #print(child.box_rect.pos, *touch.pos)
                    if child.box_rect.collide_point(*touch.pos):
                        #print("Touch collides with", child.node_id)
                        #Check for collision in the outputs of that box only
                        for j in child.input_label_circles:
                            #print(j, child.input_label_circles[j].pos)
                            if is_point_in_ellipse(touch.pos, child.input_label_circles[j].pos, (10, 10)):
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
                                        #print(lines)
                                        #print(parent.node_id, connections[parent.node_id])
                                        #print(child.node_id, connections[child.node_id])
                                        print(connections)
                                    break
                                    
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
    def __init__(self, inputs, name, node_id, outputs, **kwargs):
        super(DraggableLabel, self).__init__(**kwargs)
        self.name = name
        self.node_id = node_id
        self.inputs = inputs
        self.outputs = outputs
        
        self.text = self.node_id
        self.size_hint = (None, None)
        self.size_x = 200
        self.size = (self.size_x, 50)

        #self.mouse_widget = mouse_widget  # Reference to the MousePositionWidget
        self.prev_pos = None  # Previous position of the widget
        self.dragging = False  # Flag to track whether the label is being dragged

        with self.canvas.before:
            self.input_labels = {}
            self.input_label_circles = {}
            
            self.output_labels = {}
            self.output_label_circles = {}
            
            self.line = None  # Initialize the line object
            self.line2 = None
            
            self.trigger_lines = {}
            
            
            self.label_color = Color(0.5, 0.5, 0.5, 1)  # Set the label background color (gray in this case)
            self.label_rect = Rectangle(pos=self.pos, size=self.size)
            self.box_color = (0.3, 0.3, 0.3, 1)  # Set the box background color
            # Create a TouchableRectangle as the box_rect
            self.box_rect = TouchableRectangle(pos=self.pos, size=self.size)
            self.add_widget(self.box_rect)

            # Define the positions of the input and output circles
            self.input_circle_pos = (self.x - 3, self.y + self.height / 2 - 5)
            self.output_circle_pos = (self.right - 7, self.y + self.height / 2 - 5)

            # Draw the input and output circles
            self.input_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.output_circle_color = Color(1, 1, 1, 1)  # Circle color when not held
            self.input_circle = Ellipse(pos=self.input_circle_pos, size=(10, 10))
            self.output_circle = Ellipse(pos=self.output_circle_pos, size=(10, 10))
            
            

        self.bind(pos=self.update_rect, size=self.update_rect)
        #self.bind(on_touch_down=self.on_touch_down_box_rect)
        #self.box_rect.bind(pos=self.update_rect, size=self.update_rect)  # Bind box_rect to update_rect method
        #self.box_rect.bind(pos=self.update_rect, size=self.update_rect)  # Bind box_rect to update_rect method
        
        with self.canvas.after:
            count = 1
            for i in self.inputs:
                #print(i)
                # Add labels to the bottom box
                label = TruncatedLabel(text=f'{i}', size=(dp(len(f'{i}')*10), dp(10)))
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
    def update_rect(self, *args):
        self.label_rect.pos = self.pos
        self.label_rect.size = self.size
        
        self.box_rect.pos = (self.x, self.y - self.height)
        self.box_rect.size = (self.width, self.height)
        # Update the positions of the input and output circles
        self.input_circle_pos = (self.x - 3, self.y + self.height / 2 - 5)
        self.output_circle_pos = (self.right - 7, self.y + self.height / 2 - 5)
        self.input_circle.pos = self.input_circle_pos
        self.output_circle.pos = self.output_circle_pos
        
        
        # Position the labels in the bottom box
        #self.label1.pos = (self.x-15, self.y - self.height - 15)
        #self.label2.pos = (self.x-15, self.y - self.height - 35)
        
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

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.dragging = True  # Set dragging to True when touch is on the label
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
        if self.line:
            self.line.points = [self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5, *touch.pos]
        
        if self.node_id in connections:
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
        self.dragging = False  # Set dragging to True when touch is on the label
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
                        # Create a line connecting the output circle of the current instance to the input circle of the other instance
                        with self.canvas:
                            Color(1, 0, 0)
                            """
                            self.line2 = Line(points=[self.output_circle_pos[0] + 5, self.output_circle_pos[1] + 5,
                                                      child.input_circle_pos[0] + 5, child.input_circle_pos[1] + 5])
                            self.connection = (child.input_circle_pos[0] + 5, child.input_circle_pos[1] + 5)
                            """
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
                            
                            print(connections)
                            
                            #print(self.connection)
                            print(child.text)
                        
                        print(child.node_id)
                        #The circle collided in that id, store in connections[id]
                        print("output_circle")
                        #The id of this node, store in connections
                        print(self.node_id)
                        #The id of circle of this node, from which is detected from touch_down store in self
                        print("input_circle")
                        #Create a line globally with id number, use the points of that point and this point
                        #lines[]
                        #Link the line id to the connections
                        break
            self.canvas.remove(self.line)
            self.line = None
            self.input_circle_color.rgba = (1, 1, 1, 1)  # Gray color
            self.output_circle_color.rgba = (1, 1, 1, 1)  # Gray color
        return super(DraggableLabel, self).on_touch_up(touch)

node_init = {
    "add" : {
        "func" : None,
        "inputs" : {
            "a" : "num", 
            "b" : "num"
        },
        "outputs": {
            "c" : "num"
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
    
def generate_node(name):
    node_id = generate_node_id(name)
    print(node_id)
    nodes[node_id] = DraggableLabel(
        name = name,
        node_id = node_id,
        inputs = node_init[name]["inputs"],
        outputs = node_init[name]["outputs"])
        
"""
print(node_init["add"]["inputs"])

for i in node_init["add"]["inputs"]:
    print(i, node_init["add"]["inputs"])
"""

class DraggableLabelApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        mouse_widget = MousePositionWidget()
        
        
        #draggable_label1 = DraggableLabel()
        #draggable_label2 = DraggableLabel()
        generate_node("add")
        generate_node("add")
        generate_node("add")
        print("printing nodes")
        for i in nodes:
            print(i)
            layout.add_widget(nodes[i])
        
        layout.add_widget(mouse_widget)
        
        #layout.add_widget(draggable_label1)
        #layout.add_widget(draggable_label2)

        return layout

if __name__ == '__main__':
    DraggableLabelApp().run()
