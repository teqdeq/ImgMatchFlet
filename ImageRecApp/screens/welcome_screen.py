from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout

class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super(WelcomeScreen, self).__init__(**kwargs)

        # Create the layout for the screen
        layout = GridLayout(cols=1)

        # Add a button at the bottom middle of the screen
        button = Button(text="Enter", size_hint=(0.5, 0.2), pos_hint={'center_x': 0.5})
        button.bind(on_press=self.on_enter_pressed)
        layout.add_widget(button)

        self.add_widget(layout)

    def on_enter_pressed(self, instance):
        # Handle the button press event
        # You can switch to the matching screen here
        # For example:
        self.manager.current = "matching"
