import cv2
import numpy as np
import customtkinter
import tkinter
from PIL import Image

class ZoomingFrame(customtkinter.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, border_width=5)

        self.app = app

        self.grid_rowconfigure((0), weight=0)
        self.grid_rowconfigure((1), weight=1)
        self.grid_columnconfigure(0,weight=1)

        # Constants
        self.processedImageZoomingPIL=self.app.processedImagePIL
        self.processedImageZoomingCV2=np.copy(self.app.processedImageCV2)
        self.zoomedBitMatrix=np.copy(self.app.bitMatrix)
        self.Scaling=1
        self.zoomingOption="Nearest Neighbor Interpolation"

        # Setup Frames
        self.actionFrame = customtkinter.CTkFrame(self,fg_color="transparent",border_width=5)
        self.actionFrame.grid_columnconfigure((0,1,2), weight=1)
        self.actionFrame.grid_rowconfigure((0), weight=1)
        self.actionFrame.grid(row=0, column=0, sticky="NESW")

        self.zoomingEventOptions= customtkinter.CTkOptionMenu(self.actionFrame,values=["Nearest Neighbor Interpolation", "Bilinear Interpolation"], width= 225, dynamic_resizing=False,command=self.newOptionZoom)
        self.zoomingEventOptions.grid(row=0, column=0, pady=10)

        self.zoomingEvent_sliderLabel = customtkinter.CTkLabel(self.actionFrame, text=f"{self.Scaling:.2f}x")
        self.zoomingEvent_sliderLabel.grid(row=0, column=1, pady=10)

        self.zoomingEvent_slider = customtkinter.CTkSlider(self.actionFrame, from_=1, to=300, number_of_steps=300, command=self.zoomSliderEvent)
        self.zoomingEvent_slider.set(100)
        self.zoomingEvent_slider.grid(row=0, column=2, pady=10)

        self.imageFrame = customtkinter.CTkFrame(self,fg_color="transparent",border_width=5)
        self.imageFrame.grid_rowconfigure((0,1), weight=1)
        self.imageFrame.grid_columnconfigure((0), weight=1)
        self.imageFrame.grid(row=1, column=0, sticky="NESW")

        self.zoomingEventLabel=customtkinter.CTkLabel(self.imageFrame,text="Zoomed Image", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.zoomingEventLabel.grid(row=0, column=0, pady=10,sticky='N')

        self.zoomingEventImageLabel=customtkinter.CTkLabel(self.imageFrame,image=self.processedImageZoomingPIL,text="")
        self.zoomingEventImageLabel.grid(row=1, column=0,sticky='N')
    
    def ungridZoomingEvent(self):
        self.zoomingEventOptions.grid_forget()
        self.zoomingEvent_slider.grid_forget()
        self.zoomingEvent_sliderLabel.grid_forget()
        self.zoomingEventImageLabel.grid_forget()
        self.zoomingEventLabel.grid_forget()

    def resetZoomingVariables(self):
        self.processedImageZoomingPIL=None
        self.processedImageZoomingCV2=None
        self.zoomedBitMatrix=None
        self.Scaling=1
        self.zoomingEvent_slider.set(100)
        self.zoomingOption="Nearest Neighbor Interpolation"

    def updateInputs(self):
        self.processedImageZoomingPIL=np.copy(self.app.processedImagePIL)
        self.processedImageZoomingCV2=np.copy(self.app.processedImageCV2)
        self.zoomedBitMatrix=np.copy(self.app.bitMatrix)
        self.processedImageZoomingCV2 = self.selectOptionZoom(self.zoomingOption)
        self.updateZoomingImage()

    def updateZoomingImage(self):
        height, width = self.processedImageZoomingCV2.shape[:2]

        self.zoomingEvent_sliderLabel.configure(text=f"{self.Scaling:.2f}x")
        self.processedImageZoomingPIL = customtkinter.CTkImage(
            Image.fromarray(cv2.cvtColor(cv2.convertScaleAbs(self.processedImageZoomingCV2), cv2.COLOR_BGR2RGB)),
            size=(width, height)
        )
        
        self.zoomingEventImageLabel.configure(image=self.processedImageZoomingPIL)

        self.update_idletasks()
    
    def newOptionZoom(self,option):
        self.processedImageZoomingCV2 = self.selectOptionZoom(option)
        self.updateZoomingImage()

    def selectOptionZoom(self, option):
        self.zoomingOption=option

        if option == "Nearest Neighbor Interpolation":
            processedImageZoomingCV2 = nearest_neighbor_interpolation(self.app.processedImageCV2,self.Scaling)
        
        elif option == "Bilinear Interpolation":
            processedImageZoomingCV2 = bilinear_interpolation(self.app.processedImageCV2,self.Scaling)

        return processedImageZoomingCV2

    def zoomSliderEvent(self, value):
        self.Scaling=value/100
        self.processedImageZoomingCV2 = self.selectOptionZoom(self.zoomingOption)
        self.updateZoomingImage()

def bilinear_interpolation(bitmatrix,scale):
    height, width = bitmatrix.shape[:2]
    new_height, new_width = int(height*scale), int(width*scale)

    # Create grids for new coordinates
    x_new = np.linspace(0, width - 1, new_width)
    y_new = np.linspace(0, height - 1, new_height)

    # Create meshgrid for interpolation
    x, y = np.meshgrid(x_new, y_new)

    # Find the four nearest neighbors
    x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int) #extracts top left corner of pixel
    x2, y2 = np.minimum(x1 + 1, width - 1), np.minimum(y1 + 1, height - 1) # extracts bottom right corner of pixcels

    # Get Q Values
    top_left = bitmatrix[y1, x1]
    top_right = bitmatrix[y1, x2]
    bottom_left = bitmatrix[y2, x1]
    bottom_right = bitmatrix[y2, x2]

    frac_x = x - x1
    frac_y = y - y1

    top_interpolation = top_left * (1 - frac_x) + top_right * frac_x
    bottom_interpolation = bottom_left * (1 - frac_x) + bottom_right * frac_x

    new_matrix = top_interpolation * (1 - frac_y) + bottom_interpolation * frac_y

    return new_matrix

def nearest_neighbor_interpolation(bitmatrix,scale):
    height, width = bitmatrix.shape[:2]

    newHeight = int(height*scale)
    newWidth = int(width*scale)
    
    # Generate new coordinates using meshgrid
    new_x, new_y = np.meshgrid(np.arange(newWidth), np.arange(newHeight))

    # Calculate corresponding coordinates in the original image
    source_x = (new_x / scale).astype(int)
    source_y = (new_y / scale).astype(int)

    # Clip coordinates to stay within the original image boundaries
    source_x = np.clip(source_x, 0, width - 1)
    source_y = np.clip(source_y, 0, height - 1)

    # Use advanced indexing to get values from the original image
    new_matrix = bitmatrix[source_y, source_x]

    return new_matrix

