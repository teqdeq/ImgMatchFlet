from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from screens.welcome_screen import WelcomeScreen
from screens.matching_screen import MatchingScreen
from screens.choose_screen import ChooseScreen

class ImageMatchingApp(App):
    def build(self):
        # Create a screen manager
        screen_manager = ScreenManager()

        # Add screens to the screen manager
        screen_manager.add_widget(WelcomeScreen(name="welcome"))
        screen_manager.add_widget(MatchingScreen(name="matching"))
        screen_manager.add_widget(ChooseScreen(name="choose"))

        return screen_manager


if __name__ == "__main__":
    ImageMatchingApp().run()
