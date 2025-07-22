from Brain import Brain
import tkinter as tk

#ML methods - - - - - - - - - - - - - - -
normalizePoints = Brain.normalizePoints
genNormalMatrix = Brain.genNormalMatrix
visualizeMatrix = Brain.visualizeMatrix
# - - - - - - - - - - - - - - - - - - - -

class DrawingPad:
    def __init__(self,app):
        self.drawSize = 10              # Size of the drawn dot
        self.canvasSize = [400, 300]    # Store canvas size
        self.canvasDrawn = []
        self.canvas = tk.Canvas(        # Create canvas
            app, 
            width=self.canvasSize[0], 
            height=self.canvasSize[1], 
            bg="white"
        )                               # Bind mouseclick & release events
        self.canvas.bind("<B1-Motion>", self.draw)      
        self.canvas.bind("<ButtonRelease-1>", self.clear_canvas)      
        self.canvas.pack()              # Render canvas

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #Method for drawing on the canvas                 
    def draw(self,event):
        x, y = event.x, event.y          #Get mouse click coordinates
        r = self.drawSize                
        self.canvasDrawn.append([x,y])   #record   
        self.canvas.create_oval(         #Draw a dot
            x-r, y-r, x+r, y+r, fill="black"
        )

    #Method for clearing the canvas
    def clear_canvas(self, event):
        print(visualizeMatrix(genNormalMatrix(normalizePoints(self.canvasDrawn))))
        self.canvas.delete("all")        #Clear all graphics from canvas
        self.canvasDrawn.clear()         #Clear the recorded data