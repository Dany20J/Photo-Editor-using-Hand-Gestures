import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from tkinter import ttk
import images_operations as ops
import numpy as np
from tkinter.colorchooser import askcolor

import image_processor as ip

class State:
    def __init__(self, posX, posY, angle, scale, drawing, previous, next):
        self.posX = posX
        self.posY = posY
        self.angle = angle
        self.scale = scale
        self.drawing = drawing
        self.previous = previous
        self.next = next


class App:
    def __init__(self, window_title):
        
        self.loadedImage = False

        self.window = tkinter.Tk()

        width = self.window.winfo_screenwidth()               
        height = self.window.winfo_screenheight() 
        self.window.geometry("%dx%d" % (width, height))
        self.window.title(window_title)
        
        self.sensitivityRotateFactor = 10
        self.sensitivityScaleFactor = 100
        self.pickedColor = (255,255,255)
        

        
        self.operationsVariables = {
            "posX":0,
            "posY":0,
            "angle":0,
            "scale":self.sensitivityScaleFactor,
        }
        
        self.events = {
            "buttonPress" :"<ButtonPress-1>",
            "buttonMotion" :"<B1-Motion>",
            "buttonRelease" :"<ButtonRelease-1>",
        }
        
        self.mousePointerRect = None
        
        self.CANVAS_WIDTH = 800
        self.CANVAS_HEIGHT = 800

        self.cv_image = None
        self.sketch_image = None
        self.selected_operation = tk.StringVar(None,"translate") #translate,rotate,scale,paint

        self.paint_thickness = tk.IntVar(None,2)
        
        

        self.layout(self.window,width,height)
        self.drawingImage = np.zeros(shape = (self.CANVAS_WIDTH, self.CANVAS_HEIGHT, 3), dtype = 'uint8')

        self.state = State(
                self.operationsVariables['posX'],
                self.operationsVariables['posY'],
                self.operationsVariables['angle'],
                self.operationsVariables['scale'],
                self.drawingImage,
                None,
                None
        )
        self.saved_image = False

        self.window.mainloop()
        

    def setSelectedOperationCallback(self,value):
        self.selected_operation.set(value)

        
    def guiEventsCallback(self,event):
        if self.loadedImage == False:
            return
        
        eventType = str(event.type)
        x,y = event.x, event.y
        
        if eventType == "ButtonPress":
            self.canvas.initialCoord = x, y
            self.canvas.previousCoord = x,y

        elif eventType == "Motion":

            px, py = self.canvas.previousCoord
            self.canvas.deltaCoord = x - px, y - py
            self.canvas.previousCoord = x,y
            
            direction = x - self.canvas.initialCoord[0]
            
            if self.selected_operation.get() == 'translate':
                self.operationsVariables["posX"] += self.canvas.deltaCoord[0]
                self.operationsVariables["posY"] += self.canvas.deltaCoord[1]
                
            elif self.selected_operation.get() == 'rotate':
                angle = np.sqrt(self.canvas.deltaCoord[0]**2 + self.canvas.deltaCoord[1]**2)
                if direction < 0: angle *= -1
                self.operationsVariables["angle"] += angle
                
            elif self.selected_operation.get() == 'scale':
                scale = np.sqrt(self.canvas.deltaCoord[0]**2 + self.canvas.deltaCoord[1]**2)
                if direction < 0: scale *= -1
                self.operationsVariables["scale"] += scale
                
            elif self.selected_operation.get() == "paint":
                color = self.pickedColor
                self.drawingImage = ops.drawOnImage(self.drawingImage, x, y, px, py ,self.paint_thickness.get() ,(color[2],color[1],color[0]))


            self.applyOperations()
            
            
        elif eventType == "ButtonRelease":
            initX,initY = self.canvas.initialCoord

            temp = State(
                self.operationsVariables['posX'],
                self.operationsVariables['posY'],
                self.operationsVariables['angle'],
                self.operationsVariables['scale'],
                self.drawingImage,
                self.state,
                None
            )
            self.state.next = temp
            self.state = temp



    def update(self):
        self.gesture_manager.mainLoop()
        self.window.after(15, self.update)


    def undo(self):
        if self.state == None: return
        if self.state.previous == None: return
        print("hello")

        self.state = self.state.previous
        self.operationsVariables['posX'] = self.state.posX
        self.operationsVariables['posY'] = self.state.posY
        self.operationsVariables['angle'] = self.state.angle
        self.operationsVariables['scale'] = self.state.scale
        self.drawingImage = self.state.drawing
        self.applyOperations()

    
    def applyOperations(self):
        if self.operationsVariables["scale"] / self.sensitivityScaleFactor <= 0:
            self.operationsVariables["scale"] = 0.1 * self.sensitivityScaleFactor
            
        if self.operationsVariables["scale"] / self.sensitivityScaleFactor >= 1.5:
            self.operationsVariables["scale"] = 1.5 * self.sensitivityScaleFactor
            
        scaledImage = ops.getScaledImage(
                self.cv_image,self.operationsVariables["scale"] / self.sensitivityScaleFactor,
                self.operationsVariables["scale"] / self.sensitivityScaleFactor
            )  
        
        rotatedImage = ops.getRotatedImage(
                scaledImage,
                self.operationsVariables["angle"] // self.sensitivityRotateFactor
            ) 
        
        translatedImage = ops.getTranslatedImage(
                rotatedImage,
                self.CANVAS_WIDTH,
                self.CANVAS_HEIGHT, 
                self.operationsVariables["posY"], 
                self.operationsVariables["posX"]
            )

        result = ops.getStacked(translatedImage, self.drawingImage)
        
        # self.image_to_save = result
        
        self.updateCanvas(result)
        
    def showPointer(self,event):
        x = event.x
        y = event.y
        if self.mousePointerRect is not None:
            self.canvas.delete(self.mousePointerRect)
        self.mousePointerRect = self.canvas.create_rectangle(x,y,x+15,y+15,outline='red',width=3)
        self.canvas.tag_raise(self.mousePointerRect)
        
    def layout(self,window,width,height):
        window.columnconfigure(0, weight=9)
        window.columnconfigure(1, weight=0)
        
        left_frame = tkinter.Frame(window)
        right_frame = tkinter.Frame(window)
        left_frame.grid(column=0, row=0,sticky="NW")
        right_frame.grid(column=1, row=0, sticky="NW")

        #left frame
        
        # v = tkinter.Scrollbar(left_frame)
        self.canvas = tkinter.Canvas(left_frame, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT,background= "white")
        
        self.canvas.initialCoord = None

        self.canvas.bind('<ButtonPress-1>',self.guiEventsCallback) #initial press
        self.canvas.bind('<B1-Motion>',self.guiEventsCallback)      #motion while pressing down
        self.canvas.bind('<ButtonRelease-1>',self.guiEventsCallback) #button release
        self.canvas.bind('<Motion>',self.showPointer)
        self.canvas.bind('<Leave>',self.showPointer)

        # temp = {}
        # temp.eventType = ""
        # temp.x = 0
        # temp.y = 0
        # self.guiEventsCallback(temp)


        self.canvas.pack()




        # v.pack(side = 'right', fill = 'y')
        
        
    # right frame
        ttk.Button(right_frame, text="Activate Gestures" ,width=45, command=self.activateGestureManager).pack(pady=4)

    
    
        #loading
        ttk.Label(right_frame,text="Load:").pack(fill='x',pady=6)
        self.load_button=ttk.Button(right_frame, text="Load Image" ,width=45, command=self.loadImageCallback)
        self.save_button=ttk.Button(right_frame, text="Save Image" ,width=45, command=self.saveImageCallback)
        self.undo_button=ttk.Button(right_frame, text="Undo" ,width=45, command=self.undo)
        self.load_button.pack(pady=4)
        self.save_button.pack(pady=4)
        self.undo_button.pack(pady=4)
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x',pady=4)
        
        #operations
        ttk.Label(right_frame,text="Operations:").pack(fill='x',pady=4)
        r1 = ttk.Radiobutton(right_frame, text='translate', value='translate', variable=self.selected_operation)
        r2 = ttk.Radiobutton(right_frame, text='rotate', value='rotate', variable=self.selected_operation)
        r3 = ttk.Radiobutton(right_frame, text='scale', value='scale', variable=self.selected_operation)
        r1.pack(fill='x',pady=2)
        r2.pack(fill='x',pady=2)
        r3.pack(fill='x',pady=2)
        
        paint = ttk.Radiobutton(right_frame, text='paint', value='paint', variable=self.selected_operation)
        paint.pack(fill='x',pady=2)
        #skew frame
        
        # skewFrame = tkinter.Frame(right_frame)
        # r4 = ttk.Radiobutton(skewFrame, text='skew', value='skew', variable=self.selected_operation)
        # skew_ratio = ttk.Entry(skewFrame, textvariable=self.skew_ratio ,width= 6)
        # r4.pack(side='left')
        # skew_ratio.pack(side='left',padx=26)
        # skewFrame.pack(fill='x',pady=2)
        
        color_frame = ttk.Frame(right_frame)
        self.color_button=tkinter.Button(color_frame, text="Color",command = self.chooseColorCallback,background='#ffffff')
        self.color_button.pack(side='left',fill='x',pady=2)
        paint_thickness = ttk.Entry(color_frame, textvariable=self.paint_thickness ,width= 6)
        paint_thickness.pack(side='left',fill='x',pady=2,padx=34)
        color_frame.pack(fill='x',pady=2)

        
    def activateGestureManager(self):
        self.gesture_manager = ip.ImageProcessor()
        self.gesture_manager.setGUI(self)
        self.update()
        
        
    def chooseColorCallback(self):
        rgbAndHex = askcolor(title="Pick Color:")
        self.pickedColor = rgbAndHex[0]
        self.color_button.configure(background=rgbAndHex[1])
        # print(self.pickedColor)        
        
        
    def loadImageCallback(self):
        self.loadedImage = True
        filename = filedialog.askopenfilename()
        image = cv2.imread(filename)
        self.cv_image = image
        translatedImage = ops.getTranslatedImage(image,
            self.canvas.winfo_width(),self.canvas.winfo_height(),0,0)
        # self.image_to_save = translatedImage
        self.updateCanvas(translatedImage)

    def updateCanvas(self,image):
        # self.cv_image = image
        self.image_to_save = image
        self.canvas_image =  self.cv2tk(image)
        
        self.canvas.create_image(0,0, image = self.canvas_image, anchor = tkinter.NW)
    


    def saveImageCallback(self):
        if self.saved_image:
            return
        self.saved_image = True
        filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        if filename == None:
            return
        # filename.write()
        cv2.imwrite(filename.name,self.image_to_save)
        
        # self.canvas_image.write(filename)


    def cv2tk(self,image):
        
        return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    
        

    

