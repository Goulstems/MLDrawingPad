from Brain import Brain
import tkinter as tk

#ML methods - - - - - - - - - - - - - - -
normalizePoints = Brain.normalizePoints
genNormalMatrix = Brain.genNormalMatrix
visualizeMatrix = Brain.visualizeMatrix
predictShape = Brain.predictShape
# - - - - - - - - - - - - - - - - - - - -

class DrawingPad:
    def __init__(self,app,mode="learn"):
        self.drawSize = 10              # Size of the drawn dot
        self.canvasSize = [400, 300]    # Store canvas size
        self.canvasDrawn = []
        self.mode = mode                # Store the mode
        
        # Add label input for learning mode
        if mode == "learn":
            # Create frame for label input
            label_frame = tk.Frame(app)
            label_frame.pack(pady=5)
            
            # Add label text and entry
            tk.Label(label_frame, text="Label:").pack(side=tk.LEFT)
            self.label_entry = tk.Entry(label_frame, width=20)
            self.label_entry.pack(side=tk.LEFT, padx=5)
            self.label_entry.insert(0, "triangle")  # Default value
        
        # Position canvas based on mode
        if mode == "predict":
            # Create a spacer frame to push the canvas down
            spacer = tk.Frame(app, height=100)  # Height of learn canvas + some padding
            spacer.pack()
            
            # Add prediction label for predict mode
            prediction_frame = tk.Frame(app)
            prediction_frame.pack(pady=5)
            
            tk.Label(prediction_frame, text="Prediction:").pack(side=tk.LEFT)
            self.prediction_label = tk.Label(prediction_frame, text="Draw something...", 
                                            bg="lightgray", width=20, relief="sunken")
            self.prediction_label.pack(side=tk.LEFT, padx=5)
            
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
        normalMatrix = genNormalMatrix(normalizePoints(self.canvasDrawn))
        inputVector = Brain.flattenMatrix(normalMatrix)
        
        # Use label from entry if in learn mode, otherwise predict
        if self.mode == "learn":
            label = self.label_entry.get().strip()
            Brain.trainModel(label, inputVector)
        elif self.mode == "predict":
            prediction = predictShape(inputVector)
            self.prediction_label.config(text=prediction)
        
        print(visualizeMatrix(normalMatrix))
        self.canvas.delete("all")        #Clear all graphics from canvas
        self.canvasDrawn.clear()         #Clear the recorded data