from tk import *
from tkinter import *

def on_button_click():
    print("Button was clicked!")

# 1. Create the main window
root = Tk()

# 2. Configure window properties
root.title("Simple Tkinter App")
root.geometry("300x150")

# 3. Add widgets
label = Label(root, text="Welcome to Tkinter!")
button = Button(root, text="Press Me", command=on_button_click)

# 4. Arrange widgets
label.pack(pady=10) # Add some padding
button.pack()

# 5. Start the event loop
root.mainloop()