import tkinter as tk
from DrawingPad import DrawingPad

app = tk.Tk()                          #Create new tkinter application
canvas = DrawingPad(app)               #Create a new DrawingPad entity

app.mainloop()