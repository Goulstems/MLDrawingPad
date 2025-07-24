import tkinter as tk
from DrawingPad import DrawingPad
from Brain import Brain
import atexit

Brain.loadModel()

app = tk.Tk()                                   #Create new tkinter application
                                                #Ensure app saves model when exited
atexit.register(Brain.saveModel)

learningCanvas = DrawingPad(app, "learn")       #Create a new DrawingPad entity
predictingCanvas = DrawingPad(app, "predict")   #Create a new DrawingPad entity for prediction

app.mainloop()