import tkinter as tk
from DrawingPad import DrawingPad

app = tk.Tk()                          #Create new tkinter application
learningCanvas = DrawingPad(app, "learn")       #Create a new DrawingPad entity
predictingCanvas = DrawingPad(app, "predict")  #Create a new DrawingPad entity for prediction

app.mainloop()