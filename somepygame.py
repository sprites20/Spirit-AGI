from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST, glCullFace, GL_BACK
from kivy.graphics import RenderContext, Callback, PushMatrix, PopMatrix, \
    Color, Translate, Rotate, Mesh, UpdateNormalMatrix, BindTexture
from objloader import ObjFile
import pybullet as p

class Renderer(Widget):
    def __init__(self, **kwargs):
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.source = resource_find('simple.glsl')
        self.scene = ObjFile(resource_find("output.obj"))
        self.fall_speed = -0.1  # Set the fall speed
        self.mesh_pos = 0
        self.meshes = []
        # Existing code...
        self.meshes_data = [
            {
                'file': 'output.obj',
                'position': (-5, 5, -15),
                'rotation': (1, 0, 1, 0)
            },
            {
                'file': 'output.obj',
                'position': (0, 5, -15),
                'rotation': (1, 0, 1, 0)
            }
        ]
        self.meshes_properties = [
            {
                'file': 'output.obj',
                'position': [0, 0, -15],
                'rotation': [1, 0, 1, 0]
            },
            {
                'file': 'output.obj',
                'position': [0, 5, -15],
                'rotation': [1, 0, 1, 0]
            }
        ]
        
        self.setup_bullet()
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
    def setup_bullet(self):
        # Set up the pybullet physics simulation
        p.connect(p.DIRECT)
        p.setGravity(0, -.2, 0)  # Set the gravity
        p.setTimeStep(1.0 / 60)  # Set the simulation time step

        # Create a ground plane
        #planeId = p.loadURDF("plane.urdf")

        # Add rigid bodies for each mesh
        for i, mesh_data in enumerate(self.meshes_data):
            mesh_file = mesh_data["file"]
            position = mesh_data["position"]
            orientation = mesh_data["rotation"]

            meshId = p.loadSoftBody(mesh_file, basePosition=position, baseOrientation=orientation,
                mass=1, useMassSpring=True, useBendingSprings=True, useNeoHookean=False,
                 springElasticStiffness=0.5, springDampingStiffness=0.1, springDampingAllDirections=True
            
            )
            # Store the meshId in the meshes_data for later use
            self.meshes_data[i]["meshId"] = meshId
    def add_opposite_linear_velocities(self, mesh_index1, mesh_index2, velocity):
        """
        Add opposite linear velocities to two rigid bodies.

        :param mesh_index1: Index of the first mesh in self.meshes_data.
        :param mesh_index2: Index of the second mesh in self.meshes_data.
        :param velocity: The linear velocity to add, as a tuple (vx, vy, vz).
        """
        mesh_data1 = self.meshes_data[mesh_index1]
        mesh_data2 = self.meshes_data[mesh_index2]
        meshId1 = mesh_data1["meshId"]
        meshId2 = mesh_data2["meshId"]
        
        # Add velocity to the first mesh
        current_velocity1, _ = p.getBaseVelocity(meshId1)
        new_velocity1 = [sum(x) for x in zip(current_velocity1, velocity)]
        p.resetBaseVelocity(meshId1, new_velocity1, [0, 0, 0])
        
        # Add opposite velocity to the second mesh
        opposite_velocity = tuple(-v for v in velocity)
        current_velocity2, _ = p.getBaseVelocity(meshId2)
        new_velocity2 = [sum(x) for x in zip(current_velocity2, opposite_velocity)]
        p.resetBaseVelocity(meshId2, new_velocity2, [0, 0, 0])
    
    def apply_opposite_forces(self, mesh_index1, mesh_index2, force):
        """
        Apply opposite forces to two rigid bodies.

        :param mesh_index1: Index of the first mesh in self.meshes_data.
        :param mesh_index2: Index of the second mesh in self.meshes_data.
        :param force: The force to apply, as a tuple (fx, fy, fz).
        """
        mesh_data1 = self.meshes_data[mesh_index1]
        mesh_data2 = self.meshes_data[mesh_index2]
        meshId1 = mesh_data1["meshId"]
        meshId2 = mesh_data2["meshId"]
        
        print("Applying force to mesh 1:", force)
        p.applyExternalForce(meshId1, -1, force, (0, 0, 0), p.WORLD_FRAME)
        
        opposite_force = tuple(-f for f in force)
        print("Applying opposite force to mesh 2:", opposite_force)
        p.applyExternalForce(meshId2, -1, opposite_force, (0, 0, 0), p.WORLD_FRAME)
        
    def button_callback(self, instance):
        # Callback function for the button
        # Callback function for the button
        self.add_opposite_linear_velocities(0, 1, (2, 0, 0))  # Add opposite linear velocities to the first and second rigid bodies
        print("Opposite linear velocities added!")

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
        
        #self.meshes_data[0]["rotation"].angle += delta * 30
        #self.meshes_data[1]["rotation"].angle += delta * 30
        #self.meshes_properties[0]["position"][1] += -.2
        #self.meshes_data[0]["position"].xyz = tuple(self.meshes_properties[0]["position"])
        #self.position.xyz = (0, self.mesh_pos, -15)  # Update the Translate position
        
        # Step the pybullet simulation
        p.stepSimulation()

        # Update the positions of the meshes based on the pybullet simulation
        for mesh_data in self.meshes_data:
            meshId = mesh_data["meshId"]
            pos, orn = p.getBasePositionAndOrientation(meshId)
            #print(pos, orn)
            mesh_data["position"].xyz = pos
            mesh_data["rotation"] = orn
        
    def setup_scene(self):
        for i in self.meshes_data:
            scene = ObjFile(i["file"])
            meshes = list(scene.objects.values())
            for mesh in meshes:
                BindTexture(source='Earth_TEXTURE_CM.tga', index=1)
                Color(1, 1, 1, 1)
                PushMatrix()
                i["position"] = Translate(*i["position"])  # Adjust the y-position for each mesh
                i["rotation"] = Rotate(*i["rotation"])
                UpdateNormalMatrix()
                mesh = Mesh(
                    vertices=mesh.vertices,
                    indices=mesh.indices,
                    fmt=mesh.vertex_format,
                    mode='triangles',
                )
                PopMatrix()
                self.meshes.append(mesh)  # Store the mesh in the meshes list
class RenderScreen(Screen):
    def __init__(self, **kwargs):
        super(RenderScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(Window.width, Window.height))

        # Add the 3D renderer to the layout
        renderer = Renderer(size_hint=(1, 0.8))
        layout.add_widget(renderer)

        # Add a box on top of the renderer
        box = Button(text='Box', size_hint=(0.2, 0.1), pos_hint={'x': 0.4, 'y': 0.9})
        box.bind(on_release=renderer.button_callback)  # Bind the button's on_release event to the renderer's button_callback method
        layout.add_widget(box)

        self.add_widget(layout)

class MyScreenManager(ScreenManager):
    pass

class BoxOnTopApp(App):
    def build(self):
        Window.clearcolor = (.5, .5, .5, 1)  # Set window clear color to white
        sm = MyScreenManager()
        sm.add_widget(RenderScreen(name='main'))
        return sm

if __name__ == '__main__':
    BoxOnTopApp().run()
