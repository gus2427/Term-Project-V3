import cv2
import numpy as np
import matplotlib.pyplot as plt
import customtkinter
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from tkinter import filedialog
import os

from HistorgramCreator import createHistogramData

def ungrid_all_widgets(frame):
    for widget in frame.winfo_children():
        widget.grid_forget()

class IntensityFrame(customtkinter.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, border_width=5)

        self.app = app

        self.grid_rowconfigure((0), weight=0)
        self.grid_rowconfigure((1), weight=1)
        self.grid_columnconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=0)

        # Frame
        self.contrastActionFrame = customtkinter.CTkFrame(self,border_width=5)
        self.contrastActionFrame.columnconfigure((0,1,2,3,4),weight=1)
        self.contrastActionFrame.rowconfigure((0),weight=1)
        self.contrastActionFrame.grid(row=0,column=0, columnspan=2,sticky="nsew")

        self.contrastGraphFrame = customtkinter.CTkFrame(self,border_width=5)
        self.contrastGraphFrame.columnconfigure((0,1),weight=1)
        self.contrastGraphFrame.rowconfigure((0,1),weight=1)
        self.contrastGraphFrame.grid(row=1,column=0,sticky="nsew")

        self.contrastActionFrameExtended = customtkinter.CTkFrame(self,border_width=5)
        self.contrastActionFrameExtended.rowconfigure((0),weight=1)
        self.contrastActionFrameExtended.columnconfigure((0),weight=1)
        self.contrastActionFrameExtended.grid(row=1,column=1,sticky="nsew")

        # Constants
        self.contrastC = 1 
        self.gamma = 1

        self.ROI = np.copy(self.app.ROI)
        self.originalBitmatrix=np.copy(self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]])

        self.contrastCanvasArray=[]
        self.contrast_Axs=[]

        self.contrastCanvasArray.append(FigureCanvasTkAgg(plt.figure(tight_layout=True), master=self.contrastGraphFrame))
        self.contrast_Axs.append(plt.axes())
        
        for i in range(4):
            self.contrastCanvasArray.append(FigureCanvasTkAgg(plt.figure(figsize=(4, 4),tight_layout=True),master=self.contrastGraphFrame))
            self.contrast_Axs.append(plt.axes())

        # Main Elements
        self.negativesContrastButton = customtkinter.CTkButton(self.contrastActionFrame, text="Negatives", command=self.negativeContrastEvent)
        self.negativesContrastButton.grid(row=0,column=0,padx=10, pady=20)

        self.logarithmContrastButton = customtkinter.CTkButton(self.contrastActionFrame, text="Logarithm", command=self.logarithmContrastEvent)
        self.logarithmContrastButton.grid(row=0,column=1,padx=10, pady=20)

        self.powerContrastButton = customtkinter.CTkButton(self.contrastActionFrame, text="Power-law", command= self.powerContrastEvent)
        self.powerContrastButton.grid(row=0,column=2,padx=10, pady=20)

        self.pieceWiseContrastButton = customtkinter.CTkButton(self.contrastActionFrame, text="Piecewise-linear transformations", command=self.piecewiseContrastEvent)
        self.pieceWiseContrastButton.grid(row=0,column=3,padx=10, pady=20)

        self.histogramContrastButton = customtkinter.CTkButton(self.contrastActionFrame, text="Histogram Techniques", command= self.histogramContrastEvent)
        self.histogramContrastButton.grid(row=0,column=4,padx=10, pady=20)

        # Sub-Elements
        # Log
        self.logFrame = customtkinter.CTkFrame(self.contrastActionFrameExtended,border_width=5)
        self.logFrame.rowconfigure((0),weight=0)
        self.logFrame.rowconfigure((1),weight=0)
        self.logFrame.columnconfigure((0),weight=1)

        self.contrastCLog_SliderLabel=customtkinter.CTkLabel(self.logFrame,text=f"C = {self.contrastC:.2f}")
        self.contrastCLog_SliderLabel.grid(row=0,column=0, padx=20, sticky="S",pady=10)
        
        self.contrastCLog_Slider = customtkinter.CTkSlider(self.logFrame, from_=0, to=300, number_of_steps=300, command= self.logContrastSliderEvent)
        self.contrastCLog_Slider.set(100)
        self.contrastCLog_Slider.grid(row=1,column=0, padx=20, pady=10)

        # Power
        self.powerFrame = customtkinter.CTkFrame(self.contrastActionFrameExtended,border_width=5)
        self.powerFrame.rowconfigure((0,2),weight=0)
        self.powerFrame.rowconfigure((1,3),weight=0)
        self.powerFrame.columnconfigure((0),weight=1)

        self.contrastCPower_SliderLabel=customtkinter.CTkLabel(self.powerFrame,text=f"C = {self.contrastC:.2f}")
        self.contrastCPower_SliderLabel.grid(row=0,column=0, padx=20, pady=10)

        self.contrastCPower_Slider = customtkinter.CTkSlider(self.powerFrame, from_=0, to=300, number_of_steps=300, command=self.powerContrast_C_SliderEvent)
        self.contrastCPower_Slider.set(100)
        self.contrastCPower_Slider.grid(row=1,column=0, padx=20, pady=10)

        self.contrastGammaPowerSliderLabel =customtkinter.CTkLabel(self.powerFrame,text=f"Y = {self.gamma:.2f}")
        self.contrastGammaPowerSliderLabel.grid(row=2,column=0, padx=20, pady=10)

        self.contrastGammaPower_Slider  = customtkinter.CTkSlider(self.powerFrame, from_=0, to=300, number_of_steps=300,command=self.powerContrast_Gamma_SliderEvent)
        self.contrastGammaPower_Slider.set(150)
        self.contrastGammaPower_Slider.grid(row=3,column=0, padx=20, pady=10)

        # Piecewise
        self.piecewiseFrame = customtkinter.CTkFrame(self.contrastActionFrameExtended,border_width=5)
        self.piecewiseFrame.rowconfigure((0,2),weight=0)
        self.piecewiseFrame.rowconfigure((1),weight=1)
        self.piecewiseFrame.columnconfigure((0,1),weight=1)

        self.piecewiseActionFrame = customtkinter.CTkFrame(self.piecewiseFrame,fg_color= 'transparent')
        self.piecewiseActionFrame.rowconfigure((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19),weight=0)
        self.piecewiseActionFrame.columnconfigure((0,1),weight=1)
        self.piecewiseActionFrame.grid(row=1,column=0,columnspan=2,padx=10,pady=10, sticky= "NEWS")

        self.addPiecewiseFunction = customtkinter.CTkButton(self.piecewiseFrame,text="Add Function", command=self.addPiecewisefunction)
        self.addPiecewiseFunction.grid(row=0,column=0,columnspan=2, padx=10,pady=20)

        self.removePiecewiseFunction = customtkinter.CTkButton(self.piecewiseFrame,text="Remove Function", command=self.removePiecewisefunction)
        self.removePiecewiseFunction.grid(row=2,column=0, padx=10,pady=20)

        self.submitPiecewiseFunction = customtkinter.CTkButton(self.piecewiseFrame,text="Submit Functions", command=self.confirmFunctions)
        self.submitPiecewiseFunction.grid(row=2,column=1,padx=10,pady=20)

        # Piecewise constants
        self.gridrowBase=0
        self.functionCount=0

        self.A_labelArray = []
        self.A_slider_Array = []
        self.A_sliderVal_Array = []

        self.B_labelArray = []
        self.B_slider_Array = []
        self.B_sliderVal_Array = []

        self.m_labelArray = []
        self.m_slider_Array = []
        self.m_sliderVal_Array = []

        self.b_labelArray = []
        self.b_slider_Array = []
        self.b_sliderVal_Array = []

        self.equation_Array = []

        # Histogram
        self.exactHiso = True

        self.histogramFrame = customtkinter.CTkFrame(self.contrastActionFrameExtended,border_width=5)
        self.histogramFrame.rowconfigure((0,1,2,3),weight=1)
        self.histogramFrame.columnconfigure((0),weight=1)

        self.contrastHistogramEqualizationButton = customtkinter.CTkButton(self.histogramFrame,text="Histogram Equalization", command=self.histogramEqualization)
        self.contrastHistogramEqualizationButton.grid(row=0,column=0, padx=20, pady=10)

        self.contrastHistogramMatchingButton = customtkinter.CTkButton(self.histogramFrame,text="Histogram Matching", command=self.matchHistogram)
        self.contrastHistogramMatchingButton.grid(row=1,column=0, padx=20, pady=10)

        self.contrastHistogramExactMatchingButton = customtkinter.CTkButton(self.histogramFrame,text="Histogram Exact Matching", command=self.matchExactHistogram)
        self.contrastHistogramExactMatchingButton.grid(row=2,column=0, padx=20, pady=10)

    # Reset Functions
    def resetContrastInputs(self):
        self.contrastC=1
        self.gamma=1
        ungrid_all_widgets(self.piecewiseActionFrame)

        self.gridrowBase=0
        self.functionCount=0
        
        self.A_labelArray = []
        self.A_slider_Array = []
        self.A_sliderVal_Array = []

        self.B_labelArray = []
        self.B_slider_Array = []
        self.B_sliderVal_Array = []

        self.m_labelArray = []
        self.m_slider_Array = []
        self.m_sliderVal_Array = []

        self.b_labelArray = []
        self.b_slider_Array = []
        self.b_sliderVal_Array = []

        self.equation_Array = []
        
        self.exactHiso = True

    def updateInputs(self):
        self.ROI = np.copy(self.app.ROI)
        self.originalBitmatrix=np.copy(self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]])

    # Negative
    def negativeGraphCreator(self):
        L= 2 ** self.app.bits
        L_values=np.linspace(0,int(L-1),int(L),dtype=np.uint8)
        
        self.contrastCanvasArray[0].get_tk_widget().grid(row=0,column=0 ,rowspan = 4,columnspan=4,sticky='NEWS')
        
        self.contrast_Axs[0].clear()

        self.contrast_Axs[0].plot(L_values, L - 1 - L_values)

        self.contrast_Axs[0].set_xlabel('r_k')
        self.contrast_Axs[0].set_ylabel('s_k')
        self.contrast_Axs[0].set_title('Negative Transformation function') 

        self.contrastCanvasArray[0].draw()

    def negativeContrastEvent(self):
        ungrid_all_widgets(self.contrastGraphFrame)
        ungrid_all_widgets(self.contrastActionFrameExtended)
        self.resetContrastInputs()
        
        s = negativeTransform(self.originalBitmatrix,self.app.bits)
        self.app.stitchROIandDisplay(s)
        self.negativeGraphCreator()
   
    # Log
    def logGraphCreator(self):
        L= 2 ** self.app.bits

        L_x = np.linspace(0,int(L-1), int(L))

        L_y = np.log10(1+L_x)

        L_y = self.contrastC*scaledToBits(L_y,self.app.bits)

        L_y = clipValues(L_y,self.app.bits).astype(np.uint8)

        self.contrastCanvasArray[0].get_tk_widget().grid(row=0,column=0 ,rowspan = 4,columnspan=4,sticky='NEWS')
        
        self.contrast_Axs[0].clear()

        self.contrast_Axs[0].plot(L_x, L_y)

        self.contrast_Axs[0].set_xlabel('r_k')
        self.contrast_Axs[0].set_ylabel('s_k')
        self.contrast_Axs[0].set_title('Log Transformation function s=c*log(1+r_k)') 

        self.contrastCanvasArray[0].draw()

    def logContrastSliderEvent(self,value):
        self.contrastC=value/100
        self.contrastCLog_SliderLabel.configure(text=f"C = {self.contrastC:.2f}")

        s = logTransform(self.originalBitmatrix, self.contrastC,self.app.bits)
        self.app.stitchROIandDisplay(s)
        self.logGraphCreator()
    
    def logarithmContrastEvent(self):
        ungrid_all_widgets(self.contrastGraphFrame)
        ungrid_all_widgets(self.contrastActionFrameExtended)
        self.resetContrastInputs()

        self.logFrame.grid(row=0,column=0,sticky= 'nesw')
        s = logTransform(self.originalBitmatrix,self.contrastC,self.app.bits)
        self.app.stitchROIandDisplay(s)
        self.logGraphCreator()

    # Power
    def powerGraphCreator(self):
        L= 2 ** self.app.bits

        L_x = np.linspace(0,int(L-1), int(L))

        s_y = np.power(L_x.astype(float),self.gamma)

        s_y = self.contrastC*scaledToBits(s_y,self.app.bits)

        s_y = clipValues(s_y,self.app.bits).astype(np.uint8)

        self.contrastCanvasArray[0].get_tk_widget().grid(row=0,column=0 ,rowspan = 4,columnspan=4,sticky='NEWS')
        
        self.contrast_Axs[0].clear()

        self.contrast_Axs[0].plot(L_x, s_y)

        self.contrast_Axs[0].set_xlabel('r_k')
        self.contrast_Axs[0].set_ylabel('s_k')
        self.contrast_Axs[0].set_title('Power Transformation function s=c*r_k^y)') 

        self.contrastCanvasArray[0].draw()

    def powerContrast_C_SliderEvent(self, value):
        self.contrastC=value/100

        self.contrastCPower_SliderLabel.configure(text=f"C = {self.contrastC:.2f}")

        s = powerTransform(self.originalBitmatrix,self.contrastC,self.gamma,self.app.bits)
        self.app.stitchROIandDisplay(s)
        self.powerGraphCreator()

    def powerContrast_Gamma_SliderEvent(self,value):
        if value <= 150:
            self.gamma = pow((value-150)/150,3) + 1
        else:
            self.gamma = pow((value-150)/50,3) + 1

        self.contrastGammaPowerSliderLabel.configure(text=f"Y = {self.gamma:.2f}")

        s = powerTransform(self.originalBitmatrix,self.contrastC,self.gamma,self.app.bits)
        self.app.stitchROIandDisplay(s)
        self.powerGraphCreator()

    def powerContrastEvent(self):
        ungrid_all_widgets(self.contrastGraphFrame)
        ungrid_all_widgets(self.contrastActionFrameExtended)
        self.resetContrastInputs()

        self.powerFrame.grid(row=0,column=0,sticky= 'nesw')
        s = powerTransform(self.originalBitmatrix,self.contrastC,self.gamma,self.app.bits)
        self.app.stitchROIandDisplay(s)
        self.powerGraphCreator()

    # Piecewise
    def piecewiseContrastEvent(self):
        ungrid_all_widgets(self.contrastGraphFrame)
        ungrid_all_widgets(self.contrastActionFrameExtended)
        self.resetContrastInputs()

        self.piecewiseFrame.grid(row=0,column=0,sticky= 'nesw')

    def A_Slider_Event(self,value,index):
        self.A_sliderVal_Array[index] = value
        self.A_labelArray[index].configure(text=f"A = {self.A_sliderVal_Array[index]}")
        self.confirmFunctions()

    def B_Slider_Event(self,value,index):
        self.B_sliderVal_Array[index] = value
        self.B_labelArray[index].configure(text=f"B = {self.B_sliderVal_Array[index]}")
        self.confirmFunctions()

    def m_Slider_Event(self,value,index):
        self.m_sliderVal_Array[index] = value
        self.m_labelArray[index].configure(text=f"m = {self.m_sliderVal_Array[index]}")
        self.configureEquation(index)
        self.confirmFunctions()

    def b_Slider_Event(self,value,index):
        self.b_sliderVal_Array[index] = value
        self.b_labelArray[index].configure(text=f"b = {self.b_sliderVal_Array[index]}")
        self.configureEquation(index)
        self.confirmFunctions()

    def configureEquation(self,index):
        self.equation_Array[index].configure(text=f"S = {self.m_sliderVal_Array[index]:.2f}*r + {self.b_sliderVal_Array[index]:.2f}")

    def pieceGrapher(self):
        L= 2 ** self.app.bits

        L_x = np.linspace(0,int(L-1), int(L))

        s_y = self.extractPiecewiseFunction(L_x)

        self.contrastCanvasArray[0].get_tk_widget().grid(row=0,column=0 ,rowspan = 4,columnspan=4,sticky='NEWS')
        
        self.contrast_Axs[0].clear()

        self.contrast_Axs[0].plot(L_x, s_y)

        self.contrast_Axs[0].set_xlabel('r_k')
        self.contrast_Axs[0].set_ylabel('s_k')
        self.contrast_Axs[0].set_title('Piecewise Transformation function') 

        self.contrastCanvasArray[0].draw()

    def extractPiecewiseFunction(self, matrix):
        s = np.zeros_like(matrix,dtype=np.float64)

        Avals = np.array(self.A_sliderVal_Array)
        Bvals = np.array(self.B_sliderVal_Array)
        mvals = np.array(self.m_sliderVal_Array)
        bvals = np.array(self.b_sliderVal_Array)

        for i in range(len(Avals)):

            if Bvals[i] == 2**self.app.bits-1:
                mask = np.logical_and(Avals[i] <= matrix, matrix <= Bvals[i])
            else:
                mask = np.logical_and(Avals[i] <= matrix, matrix < Bvals[i])
                   
            s += np.where(mask,mvals[i] * matrix + bvals[i], 0)

        s = clipValues(s,self.app.bits).astype(np.uint8)

        return s

    def confirmFunctions(self):
        s = self.extractPiecewiseFunction(self.originalBitmatrix)
        self.app.stitchROIandDisplay(s)
        self.pieceGrapher()

    def addPiecewisefunction(self):
        if self.functionCount == 4:
            tkinter.messagebox.showerror(title="Error",message="Error: Reached Max function Count (4)")
            return
        L= 2 ** self.app.bits

        self.A_sliderVal_Array.append(0)
        self.A_labelArray.append(customtkinter.CTkLabel(self.piecewiseActionFrame,text=f"A = {self.A_sliderVal_Array[-1]}"))
        self.A_labelArray[-1].grid(row=self.gridrowBase,column=0,padx=15)

        self.A_slider_Array.append(customtkinter.CTkSlider(self.piecewiseActionFrame, from_=0, to=L-1, number_of_steps=L-1, command=lambda value, index=self.functionCount: self.A_Slider_Event(value,index)))
        self.A_slider_Array[-1].grid(row=self.gridrowBase+1,column=0,padx=10)
        self.A_slider_Array[-1].set(0)

        self.B_sliderVal_Array.append(0)
        self.B_labelArray.append(customtkinter.CTkLabel(self.piecewiseActionFrame,text=f"B = {self.B_sliderVal_Array[-1]}"))
        self.B_labelArray[-1].grid(row=self.gridrowBase,column=1,padx=15)

        self.B_slider_Array.append(customtkinter.CTkSlider(self.piecewiseActionFrame, from_=0, to=L-1, number_of_steps=L-1, command=lambda value, index=self.functionCount: self.B_Slider_Event(value,index)))
        self.B_slider_Array[-1].grid(row=self.gridrowBase+1,column=1,padx=10)
        self.B_slider_Array[-1].set(0)

        self.m_sliderVal_Array.append(0)
        self.m_labelArray.append(customtkinter.CTkLabel(self.piecewiseActionFrame,text=f"m = {self.m_sliderVal_Array[-1]}"))
        self.m_labelArray[-1].grid(row=self.gridrowBase+3,column=0,padx=15)

        self.m_slider_Array.append(customtkinter.CTkSlider(self.piecewiseActionFrame, from_=-15, to=15, number_of_steps=60, command=lambda value, index=self.functionCount: self.m_Slider_Event(value,index)))
        self.m_slider_Array[-1].grid(row=self.gridrowBase+4,column=0,padx=10)

        self.b_sliderVal_Array.append(0)
        self.b_labelArray.append(customtkinter.CTkLabel(self.piecewiseActionFrame,text=f"b = {self.b_sliderVal_Array[-1]}"))
        self.b_labelArray[-1].grid(row=self.gridrowBase+3,column=1,padx=15)

        self.b_slider_Array.append(customtkinter.CTkSlider(self.piecewiseActionFrame, from_=-(L-1), to=L-1, number_of_steps=2*L-2, command=lambda value, index=self.functionCount: self.b_Slider_Event(value,index)))
        self.b_slider_Array[-1].grid(row=self.gridrowBase+4,column=1,padx=10)
        
        self.equation_Array.append(customtkinter.CTkLabel(self.piecewiseActionFrame,text=f"S = {self.m_sliderVal_Array[-1]:.2f}*r+{self.b_sliderVal_Array[-1]:.2f}"))
        self.equation_Array[-1].grid(row=self.gridrowBase+2,column=0,columnspan=2,padx=10,pady=15)

        self.gridrowBase += 5
        self.functionCount +=1

    def removePiecewisefunction(self):
        if self.functionCount == 0:
            tkinter.messagebox.showerror(title="Error",message="Error: Nothing to Remove")
            return

        self.A_sliderVal_Array.pop()
        self.A_labelArray[-1].grid_forget()
        self.A_labelArray.pop()

        self.A_slider_Array[-1].grid_forget()
        self.A_slider_Array.pop()

        self.B_sliderVal_Array.pop()
        self.B_labelArray[-1].grid_forget()
        self.B_labelArray.pop()

        self.B_slider_Array[-1].grid_forget()
        self.B_slider_Array.pop()

        self.m_sliderVal_Array.pop()
        self.m_labelArray[-1].grid_forget()
        self.m_labelArray.pop()

        self.m_slider_Array[-1].grid_forget()
        self.m_slider_Array.pop()

        self.b_sliderVal_Array.pop()
        self.b_labelArray[-1].grid_forget()
        self.b_labelArray.pop()

        self.b_slider_Array[-1].grid_forget()
        self.b_slider_Array.pop()
        
        self.equation_Array[-1].grid_forget()
        self.equation_Array.pop()
        
        self.gridrowBase -= 5
        self.functionCount -=1

    # Histogram
    def histogramEqualization(self):
        self.contrastCanvasArray[2].get_tk_widget().grid(row=0,column=1,padx=10,pady=10)

        s = histogramEqualizationTransform(self.originalBitmatrix,self.app.bits)
        self.app.stitchROIandDisplay(s)
        updateContrastHistogram(self.contrastCanvasArray[2], self.contrast_Axs[2], self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]], self.app.bits)

        self.contrast_Axs[2].set_title(f'Output Histogram')  # Changed 'ax.title' to 'ax.set_title'

    def matchHistogram(self):
        if self.app.bits != 8:
            tkinter.messagebox.showerror(title="Error",message="Error: required to be in 8-bits")
            return

        currdir = os.getcwd()
        temp = filedialog.askopenfile(parent=self.app.root, initialdir=currdir, title='Please select Image to Histogram Match')
        
        if temp == None:
            return
        
        img = cv2.imread(temp.name, cv2.IMREAD_GRAYSCALE)

        self.matchingHist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])

        self.matchingPDF = self.matchingHist / np.sum(self.matchingHist)

        self.contrastCanvasArray[2].get_tk_widget().grid(row=0,column=1,padx=10,pady=10)

        s = histogramMatchingTransform(self.originalBitmatrix,self.app.bits,self.matchingPDF)
        self.app.stitchROIandDisplay(s)
        updateContrastHistogram(self.contrastCanvasArray[2], self.contrast_Axs[2], self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]], self.app.bits)
        self.contrast_Axs[2].set_title(f'Output Histogram')  # Changed 'ax.title' to 'ax.set_title'

    def matchExactHistogram(self):

        if self.exactHiso == True:
            self.matchingExactHist = np.round(np.ones(int(2**self.app.bits)) * self.originalBitmatrix.shape[0] * self.originalBitmatrix.shape[1] / (2**self.app.bits-1))
            self.exactHiso = False
        else: 
            self.matchingExactHist = np.round(np.arange(int(2**self.app.bits)) * self.originalBitmatrix.shape[0] * self.originalBitmatrix.shape[1] / (2**self.app.bits-1) * 2 / (2**self.app.bits))
            self.exactHiso = True


        self.contrastCanvasArray[2].get_tk_widget().grid(row=0,column=1,padx=10,pady=10)

        s = histogramExactMatchingTransform(self.originalBitmatrix,self.matchingExactHist, 3)
        self.app.stitchROIandDisplay(s)

        updateContrastHistogram(self.contrastCanvasArray[2], self.contrast_Axs[2], self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]], self.app.bits)
        self.contrast_Axs[2].set_title(f'Output Histogram')  # Changed 'ax.title' to 'ax.set_title'
   
    def histogramContrastEvent(self):
        ungrid_all_widgets(self.contrastGraphFrame)
        ungrid_all_widgets(self.contrastActionFrameExtended)
        self.resetContrastInputs()

        self.histogramFrame.grid(row=0,column=0,sticky= 'nesw')

        s = self.originalBitmatrix
        self.app.stitchROIandDisplay(s)

        self.contrastCanvasArray[1].get_tk_widget().grid(row=0,column=0,padx=10,pady=10)
        updateContrastHistogram(self.contrastCanvasArray[1], self.contrast_Axs[1], self.originalBitmatrix, self.app.bits)
        self.contrast_Axs[1].set_title(f'Input Image Histogram')  

def scaledToBits(Values,bits):
    L=int(2**bits)

    output = Values/np.max(Values)*(L-1)

    return output

def clipValues(Values,bits):

    output = np.clip(Values,0,int(2**bits)-1)

    return output

def negativeTransform(bitmatrix, bits):
    L = 2 ** bits

    s = L - 1 - bitmatrix

    return s

def logTransform(bitmatrix, c, bits):
    bitmatrix = bitmatrix.astype(np.float64)
    bitmatrix = np.maximum(bitmatrix, 1e-10)

    s = np.log10(1+bitmatrix)

    s = c*scaledToBits(s,bits)

    s = clipValues(s,bits).astype(np.uint8)

    return s

def powerTransform(bitmatrix, c, y, bits):
    bitmatrix = bitmatrix.astype(np.float64)
    bitmatrix = np.maximum(bitmatrix, 1e-10)

    s = np.power(bitmatrix,y)
    
    s = c*scaledToBits(s,bits)

    s = clipValues(s,bits).astype(np.uint8)

    return s

def histogramEqualizationTransform(bitmatrix, bits):
    L= int(2**bits)

    hist, bins = np.histogram(bitmatrix.flatten(), L, [0, L])

    width, height = bitmatrix.shape[:2]

    # histogram Equalization
    Pr=hist/(width*height)

    sk=np.round((L-1)*Pr.cumsum()).astype(np.uint8)

    s = bitmatrix.copy()

    for i in range(L):
        # Find the indices where intensities fall within the bin range
        indices_to_change = np.where((bitmatrix == bins[i]))

        # Set the pixels at the specified indices to the new value
        s[indices_to_change] = sk[i]

    return s

def updateContrastHistogram(canvas, ax, bitMatrix, n_bits):
    arrayQueue, nameArray, meanArray, deviationArray =createHistogramData(bitMatrix)
    bin=2**int(n_bits)

    ax.clear()

    ax.hist(arrayQueue[0], bins=bin, range=[0, bin], color=nameArray[0],histtype='step')
    ax.set_xlabel(f'{nameArray[0]} Level')  # Changed 'ax.xlabel' to 'ax.set_xlabel'
    ax.set_ylabel('Magintude')    # Changed 'ax.ylabel' to 'ax.set_ylabel'
    ax.set_title(f'{nameArray[0]} Channel Histogram')  # Changed 'ax.title' to 'ax.set_title'
        
    canvas.draw()

def CreateGmap(lst):
    result_dict = {}

    pos=-1
    while pos < len(lst)-1:
        pos+=1
        entry=lst[pos]
        j=pos
        while(j+1<len(lst)) and (entry == lst[j+1]): # makes sure elements are unique in the dictionary
            j+=1
        
        if j!=pos:
            result_dict[entry] = j//2 # if there are multiple instantces store the center one as the output
        else:
            result_dict[entry] = j
        
        pos=j

    return result_dict

def histogramMatchingTransform(bitmatrix, bits, pz):
    L= int(2**bits)

    s = histogramEqualizationTransform(bitmatrix, bits)

    Gz=np.round((L-1)*pz.cumsum()).astype(np.uint8)
    
    zqDic= CreateGmap(Gz)

    # Use list comprehension and dictionary.get for optimization
    s = [[zqDic.get(c, c) for c in row] for row in s]

    s =  np.array(s, dtype=np.uint8)

    return s

## Exact Histogram Matching
kernel1 = 1.0 / 5.0 * np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])

kernel2 = 1.0 / 9.0 * np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])

kernel3 = 1.0 / 13.0 * np.array([[0, 0, 1, 0, 0],
                                        [0, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 1, 0, 0]])

kernel4 = 1.0 / 21.0 * np.array([[0, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 0]])

kernel5 = 1.0 / 25.0 * np.array([[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1]])

kernel_mapping = {1: [kernel1],
                        2: [kernel1, kernel2],
                        3: [kernel1, kernel2, kernel3],
                        4: [kernel1, kernel2, kernel3, kernel4],
                        5: [kernel1, kernel2, kernel3, kernel4, kernel5]}

def _get_averaged_images(img, kernels):
    return np.array([signal.convolve2d(img, kernel, 'same') for kernel in kernels])

def _get_average_values_for_every_pixel(img, number_kernels):

    kernels = kernel_mapping[number_kernels]
    averaged_images = _get_averaged_images(img, kernels)
    img_size = averaged_images[0].shape[0] * averaged_images[0].shape[1]

    # shape of averaged_images: (number averaged images, height, width).
    # Reshape in a way, that one row contains all averaged values of pixel in position (x, y)
    reshaped_averaged_images = averaged_images.reshape((number_kernels, img_size))
    transposed_averaged_images = reshaped_averaged_images.transpose()
    return transposed_averaged_images

def sort_rows_lexicographically(matrix):
    # Because lexsort in numpy sorts after the last row,
    # then after the second last row etc., we have to rotate
    # the matrix in order to sort all rows after the first column,
    # and then after the second column etc.

    rotated_matrix = np.rot90(matrix)

    # TODO lexsort is very memory hungry! If the image is too big, this can result in SIG 9!
    sorted_indices = np.lexsort(rotated_matrix)
    return matrix[sorted_indices]

def histogramExactMatchingTransform(image, reference_histogram, number_kernels):
    """
        :param image: image as numpy array.
        :param reference_histogram: reference histogram as numpy array
        :param number_kernels: The more kernels you use in order to calculate average images,
                               the more likely it is, the resulting image will have the exact
                               histogram like the reference histogram
        :return: The image with the exact reference histogram.
    """
    img_size = image.shape[0] * image.shape[1]

    merged_images = np.zeros((img_size, number_kernels + 2))

        # The first column are the original pixel values.
    merged_images[:, 0] = image.reshape((img_size,))

        # The last column of this array represents the flattened image indices.
        # These indices are necessary to keep track of the pixel positions
        # after they haven been sorted lexicographically according their values.
    indices_of_flattened_image = np.arange(img_size).transpose()
    merged_images[:, -1] = indices_of_flattened_image

        # Calculate average images and add them to merged_images
    averaged_images = _get_average_values_for_every_pixel(image, number_kernels)
    for dimension in range(0, number_kernels):
        merged_images[:, dimension + 1] = averaged_images[:, dimension]

        # Sort the array according the original pixels values and then after
        # the average values of the respective pixel
    sorted_merged_images = sort_rows_lexicographically(merged_images)

        # Assign gray values according the distribution of the reference histogram
    index_start = 0
    for gray_value in range(0, len(reference_histogram)):
        index_end = int(index_start + reference_histogram[gray_value])
        sorted_merged_images[index_start:index_end, 0] = gray_value
        index_start = index_end

        # Sort back ordered by the flattened image index. The last column represents the index
    sorted_merged_images = sorted_merged_images[sorted_merged_images[:, -1].argsort()]
    new_target_img = sorted_merged_images[:, 0].reshape(image.shape)

    return new_target_img

## add if Time
def localHistogramEnhanceTransform(bitmatrix, bits):
    L= int(2**bits)

    hist, bins = np.histogram(bitmatrix.flatten(), L, [0, L])

    width, height = bitmatrix.shape[:2]

    # histogram Equalization
    Pr=hist/(width*height)

    sk=np.round((L-1)*Pr.cumsum()).astype(np.uint8)

    s = bitmatrix.copy()

    for i in range(L):
        # Find the indices where intensities fall within the bin range
        indices_to_change = np.where((bitmatrix == bins[i]))

        # Set the pixels at the specified indices to the new value
        s[indices_to_change] = sk[i]

    return s