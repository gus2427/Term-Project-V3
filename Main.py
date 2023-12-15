import customtkinter
import tkinter
import os
import pyautogui
import numpy as np
from tkinter import filedialog
from PIL import Image
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import gc

from HistorgramCreator import HistogramFrame
from Zooming import ZoomingFrame
from IntensityTransform import IntensityFrame
from Filtering import FiteringFrame
from NoiseClass import NoiseFrame

class BaseApp:
    def __init__(self, root):
        self.root = root
    
        # configure Window
        self.root.title("Term Project")
        self.root.geometry(f"{1300}x{900}")
        self.root.minsize(1200,700)

        # congigure Grid Layout
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure((0, 2), weight=0)
        self.root.grid_rowconfigure((1, 2), weight=1)
        self.root.grid_rowconfigure((0), weight=0)

        ## App Consants
        # constants that will be used by all classes
        self.imageDir=None
        self.originalImagePIL=None
        self.originalImageCV2=None
        
        self.processedImagePIL=None
        self.processedImageCV2=None

        self.bitMatrix=None
        self.bits = 8

        self.ROI = [0, 0, 0, 0]
        self.positionState=0
        self.colorState="RGB"

        ## Main App
        # Left Side Bar Frames
        self.sidebar_frame = customtkinter.CTkFrame(self.root, corner_radius=0, border_width=5)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="NESW")
        self.sidebar_frame.grid_rowconfigure((0,1,2,3,4,5,6,7), weight=1, uniform="A")
        self.sidebar_frame.grid_propagate(False)

        # Side Bar Elements
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Term Project", font=customtkinter.CTkFont(size=20, weight="bold"),width=150)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.logo_label.grid_propagate(False)

        self.select_Image_button = customtkinter.CTkButton(self.sidebar_frame, command=self.selectPictureEvent, text="Select Image")
        self.select_Image_button.grid(row=1, column=0, padx=20, pady=10)
        self.select_Image_button.grid_propagate(False)

        self.buttonColors=self.select_Image_button._fg_color

        self.Histogram_button = customtkinter.CTkButton(self.sidebar_frame, command=self.HistogramEvent, text="Calculate histogram", fg_color='gray', state="disabled")
        self.Histogram_button.grid(row=2, column=0, padx=20, pady=10)
        self.Histogram_button.grid_propagate(False)

        self.zooming_button = customtkinter.CTkButton(self.sidebar_frame, command=self.zoomingEvent, text="Zooming/Shrinking (Gray)", fg_color='gray', state="disabled", width=170)
        self.zooming_button.grid(row=3, column=0, padx=20, pady=10)
        self.zooming_button.grid_propagate(False)

        self.contrast_button = customtkinter.CTkButton(self.sidebar_frame, command=self.contrastEvent, text="Contrast/Brightness (Gray)", fg_color='gray', state="disabled", width=170)
        self.contrast_button.grid(row=4, column=0, padx=20, pady=10)
        self.contrast_button.grid_propagate(False)

        self.filter_button = customtkinter.CTkButton(self.sidebar_frame, command=self.imageFilteringEvent, text="Image Filtering (Gray)", fg_color='gray', state="disabled")
        self.filter_button.grid(row=5, column=0, padx=20, pady=10)
        self.filter_button.grid_propagate(False)

        self.noise_button = customtkinter.CTkButton(self.sidebar_frame, command=self.noiseReductionEvent, text="Noise Reduction (Gray)", fg_color='gray', state="disabled")
        self.noise_button.grid(row=6, column=0, padx=20, pady=10)
        self.noise_button.grid_propagate(False)

        self.save_button = customtkinter.CTkButton(self.sidebar_frame, command=self.saveIMGEvent, text="Save Img", fg_color='gray', state="disabled")
        self.save_button.grid(row=7, column=0, padx=20, pady=10)
        self.save_button.grid_propagate(False)

        ## Right Side Bar
        # Gray and Color Buttons
        self.colorButtonFrame= customtkinter.CTkFrame(self.root,height=50)
        self.colorButton = customtkinter.CTkButton(self.colorButtonFrame, command=self.rgbColorEvent, text="Color", fg_color='Gray', state="disabled")
        self.grayButton = customtkinter.CTkButton(self.colorButtonFrame, command=self.greyColorEvent, text="Gray")
        self.resetButton = customtkinter.CTkButton(self.colorButtonFrame, command=self.resetApplication, text="Reset")
        self.RioButton = customtkinter.CTkButton(self.colorButtonFrame, command=self.openROIWindow, text="ROI Window")

        self.colorButton.grid(row=0,column=0, padx=10, pady=10)
        self.grayButton.grid(row=0,column=1, padx=10, pady=10)
        self.resetButton.grid(row=0,column=2, padx=10, pady=10)
        self.RioButton.grid(row=0,column=3, padx=10, pady=10)

        # Original Image
        # Frame
        self.originalImageFrame = customtkinter.CTkFrame(self.root,width=450)
        self.originalImageFrame.grid(row=1, column=2, rowspan=1, sticky="nsew")
        self.originalImageFrame.pack_propagate(False)

        # Labels
        self.ImageLabel1=customtkinter.CTkLabel(self.originalImageFrame,text="", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.ImageLabel1.pack(pady=10)

        self.originalImageLabel=customtkinter.CTkLabel(self.originalImageFrame,text="")
        self.originalImageLabel.pack(pady=10)

        # Alterated Image
        # Frame
        self.processedImageFrame = customtkinter.CTkFrame(self.root,width=450)
        self.processedImageFrame.grid(row=2, column=2, rowspan=1, sticky="nsew")
        self.processedImageFrame.pack_propagate(False)

        # Label
        self.ImageLabel2=customtkinter.CTkLabel(self.processedImageFrame,text="", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.ImageLabel2.pack(pady=10)

        self.processeingLabel=customtkinter.CTkLabel(self.processedImageFrame,text="")
        self.processeingLabel.pack(pady=10)

        ## Action Frame
        # This is where the indiviual Class Frames will be place on top of
        # it will be in the middle of the screen
        self.actionFrame = customtkinter.CTkFrame(self.root)
        self.actionFrame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.actionFrame.grid_columnconfigure((0), weight=1)
        self.actionFrame.grid_rowconfigure((0), weight=1)

        ## Set all Frames to None
        self.Histogram_Frame = None
        self.Zooming_Frame = None
        self.Intensity_Frame = None
        self.Filter_Frame = None
        self.Noise_frame = None

        ## select ROI
        self.roi_window = None

    #
    def selectPictureEvent(self):
        currdir = os.getcwd()
        temp = filedialog.askopenfile(parent=self.root, initialdir=currdir, title='Please select Image to Proccess')

        if temp !=None:
            if self.imageDir != None:
                self.resetApplication()
                self.imageDir = temp
                self.imageSetup()
                
            elif self.imageDir == None:
                self.imageDir = temp

                self.Histogram_button.configure(fg_color=self.buttonColors, state="normal")
                self.save_button.configure(fg_color=self.buttonColors, state="normal")

                self.colorButtonFrame.grid(row=0, column=2, rowspan=1, sticky="n")

                self.actionFrame.configure(border_width=5)
                self.originalImageFrame.configure(border_width=5)
                self.processedImageFrame.configure(border_width=5)
                self.colorButtonFrame.configure(border_width=5)

                self.ImageLabel1.configure(text="Original Image")
                self.ImageLabel2.configure(text="Processed Image")

                self.imageSetup()

    def imageSetup(self):
        img=cv2.imread(self.imageDir.name)
                
        height, width, channels = img.shape

        maxValue=640
        self.processedImageFrame.configure(width = maxValue+15)
        self.originalImageFrame.configure(width = maxValue+15)
        self.colorButtonFrame.configure(width = maxValue+15)
        
        if width > maxValue or height > maxValue:
            if width>height:
                ratio=maxValue/width
                height=round(ratio*height/10)*10
                img=cv2.resize(img,(maxValue,height))
            else:
                ratio=maxValue/height
                width=round(ratio*width/10)*10
                img=cv2.resize(img,(width,maxValue))

        height, width, channels = img.shape

        self.ROI[2] = width
        self.ROI[3] = height

        self.originalImagePIL=customtkinter.CTkImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),size= (width,height))
        self.originalImageCV2=img
        self.bitMatrix = np.copy(self.originalImageCV2)

        self.processedImageCV2 = np.copy(self.originalImageCV2)
        
        self.originalImageLabel.configure(image=self.originalImagePIL)
        self.processeingLabel.configure(image=self.originalImagePIL)
  
    def rgbColorEvent(self):

        if self.positionState==0 and self.colorState=="Grey":
            self.disableSideButtons()
            self.processedImageCV2=np.copy(self.originalImageCV2)
            self.bitMatrix = np.copy(self.processedImageCV2)
            self.updateProcessedLabel(self.processedImageCV2)
            self.colorState="RGB"
            self.colorButton.configure(fg_color='Gray', state="disabled")
            self.grayButton.configure(fg_color=self.buttonColors, state="normal")

        elif self.positionState==1:
            if len(self.bitMatrix.shape) != 3:
                self.disableSideButtons()
                self.colorState="RGB"

                self.processedImageCV2=np.copy(self.originalImageCV2)
                self.bitMatrix = np.copy(self.originalImageCV2)
                self.Histogram_Frame.updateInputs()

                self.Histogram_Frame.colorHistogramEvent()
                self.colorButton.configure(fg_color='Gray', state="disabled")
                self.grayButton.configure(fg_color=self.buttonColors, state="normal")
        else:
            tkinter.messagebox.showerror(title="Error",message="Error: Cannot go into Color in this selection")
    
    def greyColorEvent(self):
        self.colorButton.configure(fg_color=self.buttonColors, state="normal")
        self.grayButton.configure(fg_color='Gray', state="disabled")

        if self.positionState==0 and self.colorState=="RGB":
            self.processedImageCV2=np.copy(cv2.cvtColor(self.originalImageCV2,cv2.COLOR_BGR2GRAY))
            self.bitMatrix = np.copy(self.processedImageCV2)
            self.updateProcessedLabel(self.processedImageCV2)

        elif self.positionState==1 and self.colorState=="RGB":
            self.processedImageCV2=np.copy(cv2.cvtColor(self.originalImageCV2,cv2.COLOR_BGR2GRAY))
            self.bitMatrix = np.copy(self.processedImageCV2)
            self.Histogram_Frame.updateInputs()
            self.Histogram_Frame.greyHistogramEvent()

        self.colorState="Grey"
        self.enableSideButtons()

    # General functions
    def openROIWindow(self):
        if self.roi_window is None or not self.roi_window.winfo_exists():
            self.roi_window =  customtkinter.CTkToplevel(self.root)
            self.ROIApp = ROIWindow(self.roi_window,self)
        self.roi_window.focus()

    def stitchROIandDisplay(self,output):

        if len(output.shape)==3:
            self.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2],:] = output
            self.processedImageCV2[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2],:] = self.bitmatrix_to_CV2()
        else:
            self.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]] = output
            self.processedImageCV2[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]] = self.bitmatrix_to_CV2()

        self.updateProcessedLabel(self.processedImageCV2)

    def enableSideButtons(self):
        self.Histogram_button.configure(fg_color=self.buttonColors, state="normal")
        self.zooming_button.configure(fg_color=self.buttonColors, state="normal")
        self.contrast_button.configure(fg_color=self.buttonColors, state="normal")
        self.filter_button.configure(fg_color=self.buttonColors, state="normal")
        self.noise_button.configure(fg_color=self.buttonColors, state="normal")
        self.save_button.configure(fg_color=self.buttonColors, state="normal")

    def disableSideButtons(self):
        self.zooming_button.configure(fg_color="gray", state="disabled")
        self.contrast_button.configure(fg_color="gray", state="disabled")
        self.filter_button.configure(fg_color="gray", state="disabled")
        self.noise_button.configure(fg_color="gray", state="disabled")

    def updateProcessedLabel(self,cvImage):
        height, width = cvImage.shape[:2]

        self.processedImagePIL=customtkinter.CTkImage(Image.fromarray(cv2.cvtColor(cvImage,cv2.COLOR_BGR2RGB)),size= (width,height))

        self.processeingLabel.configure(image=self.processedImagePIL)

    def bitmatrix_to_CV2(self, matrix=None):
        scale = 255 / (2 ** self.bits - 1)

        if len(self.bitMatrix.shape) == 3:
            output = (np.round(self.bitMatrix[self.ROI[1]:self.ROI[3], self.ROI[0]:self.ROI[2], :] * scale)).astype(np.uint8)
            return output
        else:
            output = (np.round(self.bitMatrix[self.ROI[1]:self.ROI[3], self.ROI[0]:self.ROI[2]] * scale)).astype(np.uint8)
            return output
        
    # Reset Functions
    def ungrid_all_widgets(self, frame):
        for widget in frame.winfo_children():
            widget.grid_forget()

    def resetApplication(self):
        self.processedImagePIL=self.originalImagePIL
        self.processedImageCV2=np.copy(self.originalImageCV2)

        self.bitMatrix=np.copy(self.originalImageCV2)
        self.bits = 8

        self.ROI = [0, 0, 0, 0]
        height, width = self.bitMatrix.shape[:2]

        self.ROI[2] = width
        self.ROI[3] = height

        self.positionState=0
        self.colorState="RGB"

        self.colorButton.configure(fg_color='Gray', state="disabled")
        self.grayButton.configure(fg_color=self.buttonColors, state="normal")
        
        self.disableSideButtons()

        self.updateProcessedLabel(self.processedImageCV2)
        self.ungrid_all_widgets(self.actionFrame)

        plt.close('all')

        del self.Histogram_Frame
        del self.Zooming_Frame
        del self.Intensity_Frame
        del self.Filter_Frame
        del self.Noise_frame

        self.Histogram_Frame = None
        self.Zooming_Frame = None
        self.Intensity_Frame = None
        self.Filter_Frame = None
        self.Noise_frame = None

        self.roi_window = None

        gc.collect()
        
    ## Histogram Functions (position State = 1)
    def HistogramEvent(self):
        if self.positionState == 1:
            return

        self.ungrid_all_widgets(self.actionFrame)

        if self.Histogram_Frame == None:
            self.Histogram_Frame = HistogramFrame(self.actionFrame,self)

        self.positionState=1
        self.Histogram_Frame.grid(row=0,column=0, sticky="nsew")
        ## Rerun inital hitogram functions:
        if self.colorState == "Grey":
            self.Histogram_Frame.greyHistogramEvent()
        else:
            self.Histogram_Frame.colorHistogramEvent()
 
    ## Zooming Functions (position State = 2)
    def zoomingEvent(self):
        if self.colorState != "Grey":
            tkinter.messagebox.showerror(title="Error",message="Error: required To be in GrayScale")
            return
        
        if self.positionState == 2:
            return

                    
        if self.Zooming_Frame == None:
            self.Zooming_Frame = ZoomingFrame(self.actionFrame,self)
        else:
            self.Zooming_Frame.updateInputs()

        self.ungrid_all_widgets(self.actionFrame)
        self.positionState = 2
        self.Zooming_Frame.grid(row=0,column=0, sticky="nsew")

    ## Contrast Functions (position State = 3)
    def contrastEvent(self):
        if self.colorState != "Grey":
            tkinter.messagebox.showerror(title="Error",message="Error: required To be in GrayScale")
            return

        if self.positionState ==3:
            return

        self.ungrid_all_widgets(self.actionFrame)

        if self.Intensity_Frame == None:
            self.Intensity_Frame = IntensityFrame(self.actionFrame,self)
        else:
            self.Intensity_Frame.updateInputs()

        self.positionState=3
        self.ungrid_all_widgets(self.Intensity_Frame.contrastGraphFrame)
        self.ungrid_all_widgets(self.Intensity_Frame.contrastActionFrameExtended)
        self.Intensity_Frame.grid(row=0,column=0, sticky="nsew")

    ## Filter Functions (position State = 4)
    def imageFilteringEvent(self):
        if self.colorState != "Grey":
            tkinter.messagebox.showerror(title="Error",message="Error: required To be in GrayScale")
            return

        if self.positionState == 4:
            return
        
        self.ungrid_all_widgets(self.actionFrame)

        if self.Filter_Frame == None:
            self.Filter_Frame = FiteringFrame(self.actionFrame,self)
        else:
            self.Filter_Frame.updateInputs()

        self.positionState=4
        self.ungrid_all_widgets(self.Filter_Frame.GraphFrame)
        self.ungrid_all_widgets(self.Filter_Frame.ActionFrameExtended)
        self.Filter_Frame.grid(row=0,column=0, sticky="nsew")

    ## Noise Functions (position State = 5)
    def noiseReductionEvent(self):
        if self.colorState != "Grey":
            tkinter.messagebox.showerror(title="Error",message="Error: required To be in GrayScale")
            return        

        if self.positionState == 5:
            return
        
        self.ungrid_all_widgets(self.actionFrame)

        if self.Noise_frame == None:
            self.Noise_frame = NoiseFrame(self.actionFrame,self)
        else:
            self.Noise_frame.updateInputs()

        self.positionState = 5

        self.ungrid_all_widgets(self.Noise_frame.GraphFrame)
        self.ungrid_all_widgets(self.Noise_frame.ActionFrameExtended)

        self.Noise_frame.grid(row=0,column=0, sticky="nsew")

    ## Saving Image
    def saveIMGEvent(self):
        currdir = os.getcwd()
        filedir = filedialog.askdirectory(parent=self.root, initialdir=currdir, title='Please select where to save the Image')

        if self.Zooming_Frame != None:
            self.Zooming_Frame.updateInputs()
            output = self.Zooming_Frame.selectOptionZoom(self.Zooming_Frame.zoomingOption)
        else:
            output = self.processedImageCV2

        file_name, file_extension = os.path.splitext(os.path.basename(self.imageDir.name))
        cv2.imwrite(filedir+f"\{file_name}_output.jpg",output)

        tkinter.messagebox.showinfo("Notification", "Image is saved!")

class ROIWindow:
    def __init__(self, root, app):
        self.app = app
        self.root = root
        
        self.root.title("ROI Window")
        self.root.geometry(f"{1300}x{700}")
        self.root.minsize(1300,700)

        # congigure Grid Layout
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure((0), weight=0)
        self.root.grid_rowconfigure((0, 1), weight=1)

        # Constants
        self.IMG = self.app.bitMatrix
        self.ROI = self.IMG

        H,W = self.IMG.shape[:2]
        self.x1 = self.app.ROI[0] 
        self.y1 = self.app.ROI[1]
        self.x2 = self.app.ROI[2] 
        self.y2 = self.app.ROI[3]

        self.Display = self.cv2_to_IMG(self.IMG[self.y1:self.y2,self.x1:self.x2])

        # Actions Frame
        self.ActionFrame = customtkinter.CTkFrame(self.root, corner_radius=0, border_width=5)
        self.ActionFrame.grid(row=0, column=0, rowspan=2, sticky="NESW")
        self.ActionFrame.grid_rowconfigure((0,1,2,3,4,5,6), weight=1)
        self.ActionFrame.grid_columnconfigure((0,1), weight=1)

        self.x1_SliderLabel=customtkinter.CTkLabel(self.ActionFrame,text=f"x1 = {self.x1}")
        self.x1_SliderLabel.grid(row=0,column=0,padx=10, pady=20)
        self.x1_Slider = customtkinter.CTkSlider(self.ActionFrame, from_=0, to=W, number_of_steps=W, command=self.x1_sliderEvent)
        self.x1_Slider.grid(row=1,column=0,padx=10, pady=20)
        self.x1_Slider.set(self.x1)

        self.y1_SliderLabel=customtkinter.CTkLabel(self.ActionFrame,text=f"y1 = {self.y1}")
        self.y1_SliderLabel.grid(row=0,column=1,padx=10, pady=20)
        self.y1_Slider = customtkinter.CTkSlider(self.ActionFrame, from_=0, to=H, number_of_steps=H, command=self.y1_sliderEvent)
        self.y1_Slider.grid(row=1,column=1,padx=10, pady=20)
        self.y1_Slider.set(self.y1)

        self.x2_SliderLabel=customtkinter.CTkLabel(self.ActionFrame,text=f"x2 = {self.x2}")
        self.x2_SliderLabel.grid(row=2,column=0,padx=10, pady=20)
        self.x2_Slider = customtkinter.CTkSlider(self.ActionFrame, from_=0, to=W, number_of_steps=W, command=self.x2_sliderEvent)
        self.x2_Slider.grid(row=3,column=0,padx=10, pady=20)
        self.x2_Slider.set(self.x2)

        self.y2_SliderLabel=customtkinter.CTkLabel(self.ActionFrame,text=f"y2 = {self.y2}")
        self.y2_SliderLabel.grid(row=2,column=1,padx=10, pady=20)
        self.y2_Slider = customtkinter.CTkSlider(self.ActionFrame, from_=0, to=H, number_of_steps=H, command=self.y2_sliderEvent)
        self.y2_Slider.grid(row=3,column=1,padx=10, pady=20)
        self.y2_Slider.set(self.y2)

        self.resetButton = customtkinter.CTkButton(self.ActionFrame, text="Reset ROI", command=self.resetROI)
        self.resetButton.grid(row = 4, column = 0, padx = 10, pady = 10)

        self.submitButton = customtkinter.CTkButton(self.ActionFrame, text="Submit ROI", command=self.sumbitROI)
        self.submitButton.grid(row = 4, column = 1, padx = 10, pady = 10)

        # Image Frame
        self.ROIFrame = customtkinter.CTkFrame(self.root, corner_radius=0, border_width=5)
        self.ROIFrame.grid(row=0, column=1, rowspan=2, sticky="NESW")
        self.ROIFrame.grid_rowconfigure((0), weight=1)
        self.ROIFrame.grid_columnconfigure((0), weight=1)

        self.DisplayLabel=customtkinter.CTkLabel(self.ROIFrame, image=self.Display,text="")
        self.DisplayLabel.grid(row=0,column=0, padx= 10, pady=10)

    def x1_sliderEvent(self,value):
        self.x1 = int(value)
        self.x1_SliderLabel.configure(text=f"x1 = {self.x1}")
        self.setROI()

    def y1_sliderEvent(self,value):
        self.y1 = int(value)
        self.y1_SliderLabel.configure(text=f"y1 = {self.y1}")
        self.setROI()

    def x2_sliderEvent(self,value):
        self.x2 = int(value)
        self.x2_SliderLabel.configure(text=f"x2 = {self.x2}")    
        self.setROI()
       
    def y2_sliderEvent(self,value):
        self.y2 = int(value)
        self.y2_SliderLabel.configure(text=f"y2 = {self.y2}")
        self.setROI()

    def cv2_to_IMG(self,cvImage):
        height, width = cvImage.shape[:2]

        return customtkinter.CTkImage(Image.fromarray(cv2.cvtColor(cvImage,cv2.COLOR_BGR2RGB)),size= (width,height))

    def updateDisplay(self,img):
        self.DisplayLabel.configure(image=img)

    def setROI(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return
        
        self.ROI = self.IMG[self.y1:self.y2,self.x1:self.x2]

        self.updateDisplay(self.cv2_to_IMG(self.ROI))

        self.ActionFrame.update_idletasks()
        self.ROIFrame.update_idletasks()

    def resetSliders(self):
        self.y2, self.x2 = self.IMG.shape[:2]
        self.x1 = 0
        self.y1 = 0

        self.x1_SliderLabel.configure(text=f"x1 = {self.x1}")
        self.x1_Slider.set(self.x1)

        self.y1_SliderLabel.configure(text=f"y1 = {self.y1}")
        self.y1_Slider.set(self.y1)

        self.x2_SliderLabel.configure(text=f"x2 = {self.x2}")    
        self.x2_Slider.set(self.x2)
       
        self.y2_SliderLabel.configure(text=f"y2 = {self.y2}")
        self.y2_Slider.set(self.y2)

    def resetROI(self):
        self.IMG = self.app.bitMatrix
        self.ROI = self.IMG
        self.Display = self.cv2_to_IMG(self.IMG)

        self.updateDisplay(self.cv2_to_IMG(self.ROI))

        self.resetSliders()

    def sumbitROI(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            tkinter.messagebox.showerror(title="Error",message="Error: Invalid Region of Intrest. X2<=X1 or Y2<=Y1")
            self.app.roi_window.focus()
            return
        
        self.app.ROI = [self.x1,self.y1,self.x2,self.y2]
        self.app.roi_window = None

        if self.app.Histogram_Frame != None:
            self.app.Histogram_Frame.updateInputs()       
        if self.app.Intensity_Frame != None:
            self.app.Intensity_Frame.updateInputs()
        if self.app.Filter_Frame != None:
            self.app.Filter_Frame.updateInputs()
        if self.app.Noise_frame != None:
            self.app.Noise_frame.updateInputs()

        self.root.destroy()

if __name__ == "__main__":
    root = customtkinter.CTk()
    app = BaseApp(root)
    root.mainloop() 