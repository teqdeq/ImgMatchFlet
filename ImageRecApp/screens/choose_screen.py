import csv
from kivy.uix.button import Button

class ChooseScreen(Screen):
    def __init__(self, **kwargs):
        super(ChooseScreen, self).__init__(**kwargs)

    def load_entries_from_csv(self, csv_path):
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_name = row['Image Name']
                description = row['Description']
                location = row['Location']

                button = Button(text=f"{image_name}\n{description}\n{location}")
                self.ids.button_container.add_widget(button)
