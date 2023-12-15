import cv2
import numpy as np
import matplotlib.pyplot as plt
import customtkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from grayScaleNBits import convert_to_nbit_grayscale
from RGBscaledNbits import convert_to_nbit_RGB

class HistogramFrame(customtkinter.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, border_width=5)

        self.app = app

        self.grid_columnconfigure((0,1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)
        
        # Image Frames
        self.plotRFrame=customtkinter.CTkFrame(self,fg_color="transparent",border_width=5)
        self.plotRFrame.columnconfigure((0),weight=1)
        self.plotRFrame.rowconfigure((0,1),weight=1)
        self.plotRFrame.grid(row=3, column=0, rowspan=1, sticky="nsew")

        self.plotGFrame=customtkinter.CTkFrame(self,fg_color="transparent",border_width=5)
        self.plotGFrame.columnconfigure((0),weight=1)
        self.plotGFrame.rowconfigure((0,1),weight=1)
        self.plotGFrame.grid(row=3, column=1, rowspan=1, sticky="nsew")

        self.plotBFrame=customtkinter.CTkFrame(self,fg_color="transparent",border_width=5)
        self.plotBFrame.columnconfigure((0),weight=1)
        self.plotBFrame.rowconfigure((0,1),weight=1)
        self.plotBFrame.grid(row=3, column=2, rowspan=1, sticky="nsew")

        # Labels and options
        self.gridlevel_label = customtkinter.CTkLabel(self, text="# of levels: "+str(int(2**self.app.bits)), font=customtkinter.CTkFont(size=20))
        self.gridlevel_label.grid(row=0, column=1, pady=10)

        self.gridlevel_slider = customtkinter.CTkSlider(self, from_=1, to=8, number_of_steps=7, command=self.updateLevels)
        self.gridlevel_slider.grid(row=1, column=1, pady=20)
        self.gridlevel_slider.set(8)

        self.mean_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=20))
        self.mean_label.grid(row=2, column=0, pady=10)

        self.deviation_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=20))
        self.deviation_label.grid(row=2, column=2, pady=20)

        # Constants
        self.originalImageCV2 = np.copy(self.app.originalImageCV2[self.app.ROI[1]:self.app.ROI[3],self.app.ROI[0]:self.app.ROI[2]])

        self.frameArray=[self.plotBFrame,self.plotGFrame,self.plotRFrame]
        self.histogram_AXs=[]
        self.canvasHistogramArray=[]
        self.meanArray=[]
        self.deviationArray=[]

        for i in range(3):
            self.canvasHistogramArray.append(FigureCanvasTkAgg(plt.figure(figsize=(3,3),tight_layout=True),master=self.frameArray[i]))
            self.histogram_AXs.append(plt.axes())

    def updateInputs(self):
        self.originalImageCV2 = np.copy(self.app.originalImageCV2[self.app.ROI[1]:self.app.ROI[3],self.app.ROI[0]:self.app.ROI[2],:])
        
    def colorHistogramEvent(self):
        #Colored Images
        #convert get Bit Matrix
        temp, s = convert_to_nbit_RGB(self.originalImageCV2, self.app.bits)

        self.app.stitchROIandDisplay(s)

        self.meanArray, self.deviationArray = updateHistogram(self.canvasHistogramArray, self.histogram_AXs, s, self.app.bits)

        self.updateHistogramLabels()

        for i in range(len(self.canvasHistogramArray)):
            self.canvasHistogramArray[i].get_tk_widget().grid(row=0,column=0,padx=10,pady=10)

    def greyHistogramEvent(self):
        #Turn images into Gray scale and Calculate Histogram
        #Gray Images
        #convert get Bit Matrix
        temp, s = convert_to_nbit_grayscale(self.originalImageCV2, self.app.bits)

        self.app.stitchROIandDisplay(s)

        self.meanArray, self.deviationArray = updateHistogram(self.canvasHistogramArray, self.histogram_AXs, s, self.app.bits)

        self.updateHistogramLabels()

        for i in range(len(self.canvasHistogramArray)):
            if i == 1:
                self.canvasHistogramArray[i].get_tk_widget().grid(row=0,column=0,padx=10,pady=10)
            else:
                self.canvasHistogramArray[i].get_tk_widget().grid_forget()
        
    def ungridHistogramEvent(self):
        self.gridlevel_label.grid_forget()
        self.gridlevel_slider.grid_forget()
        self.mean_label.grid_forget()
        self.deviation_label.grid_forget()

        for i in self.canvasHistogramArray:
            i.get_tk_widget().grid_forget()

        for i in self.frameArray:
            i.grid_forget()

    def resetHistogramVariables(self):
        self.histogram_AXs=[]
        self.canvasHistogramArray=[]
        self.meanArray=[]
        self.deviationArray=[]
        self.gridlevel_label.configure(text="# of levels: "+str(int(2**self.app.bits)))
        self.gridlevel_slider.set(8)

    def updateLevels(self,value):
        self.app.bits=value
        self.gridlevel_label.configure(text="# of levels: "+str(int(2**self.app.bits)))
        
        if len(self.app.bitMatrix.shape) == 3:
            temp, s = convert_to_nbit_RGB(self.originalImageCV2, self.app.bits)
            
            self.app.stitchROIandDisplay(s)

        elif len(self.app.bitMatrix.shape) == 2:
            temp, s = convert_to_nbit_grayscale(self.originalImageCV2, self.app.bits)
            
            self.app.stitchROIandDisplay(s)

        self.meanArray, self.deviationArray = updateHistogram(self.canvasHistogramArray, self.histogram_AXs, s, self.app.bits)

        self.updateHistogramLabels()

    def updateHistogramLabels(self):
        if len(self.meanArray)==1:
            #gray
            self.mean_label.configure(text="Mean= "+str(round(self.meanArray[0],2)))
            self.deviation_label.configure(text="STD= "+str(round(self.deviationArray[0],2)))
        elif len(self.meanArray)==3:
            #color
            self.mean_label.configure(text="Mean: R="+str(round(self.meanArray[2],2))+" G="+str(round(self.meanArray[1],2))+" B="+str(round(self.meanArray[0],2)))
            self.deviation_label.configure(text="STD: R="+str(round(self.deviationArray[2],2))+" G="+str(round(self.deviationArray[1],2))+" B="+str(round(self.deviationArray[0],2)))
        else:
            raise ValueError("Input image has an unsupported number of dimensions.")


# Global Functions
def historgramStatistics(arrayQueue):
    ## Need to a nth moment maybe % global and locals
    meanArray=[]
    deviationArray=[]

    for i in arrayQueue:
        meanArray.append(np.mean(i))
        deviationArray.append(np.std(i))
    
    return meanArray, deviationArray 

def createHistogramData(bitMatrix):
    if len(bitMatrix.shape) == 3:
        cols, rows, channels = bitMatrix.shape
        nameArray = ["Blue", "Green", "Red"]
        arrayQueue = [bitMatrix[:, :, i].flatten() for i in range(channels)]
    elif len(bitMatrix.shape) == 2:
        channels = 1
        nameArray = ["Gray"]
        arrayQueue = [bitMatrix.flatten()]
    else:
        raise ValueError("Input image has an unsupported number of dimensions.")
    
    meanArray, deviationArray = historgramStatistics(arrayQueue)

    return arrayQueue, nameArray, meanArray, deviationArray

def updateHistogram(canvas, ax, bitMatrix, n_bits):
    arrayQueue, nameArray, meanArray, deviationArray =createHistogramData(bitMatrix)
    bin=2**int(n_bits)

    if len(arrayQueue)==1:
        ax[1].clear()

        ax[1].hist(arrayQueue[0], bins=bin, range=[0, bin], color=nameArray[0],histtype='step')
        ax[1].set_xlabel(f'{nameArray[0]} Level')  # Changed 'ax.xlabel' to 'ax.set_xlabel'
        ax[1].set_ylabel('Magintude')    # Changed 'ax.ylabel' to 'ax.set_ylabel'
        ax[1].set_title(f'{nameArray[0]} Channel Histogram')  # Changed 'ax.title' to 'ax.set_title'
        
        canvas[1].draw()
    else:
        for i in range(len(arrayQueue)):
            ax[i].clear()

            ax[i].hist(arrayQueue[i], bins=bin, range=[0, bin], color=nameArray[i],histtype='step')
            ax[i].set_xlabel(f'{nameArray[i]} Level')  # Changed 'ax.xlabel' to 'ax.set_xlabel'
            ax[i].set_ylabel('Magintude')    # Changed 'ax.ylabel' to 'ax.set_ylabel'
            ax[i].set_title(f'{nameArray[i]} Channel Histogram')  # Changed 'ax.title' to 'ax.set_title'
            
            canvas[i].draw()
    
    return meanArray, deviationArray 


def createHistogram(bitMatrix, n_bits):
    bin = 2 ** int(n_bits)

    arrayQueue, nameArray, meanArray, deviationArray = createHistogramData(bitMatrix) 

    figureArray = []
    axArray = []

    for i in range(len(arrayQueue)):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(arrayQueue[i], bins=bin, range=[0, bin], color=nameArray[i], histtype='step')
        ax.set_xlabel(f'{nameArray[i]} Level') 
        ax.set_ylabel('Magintude')   
        ax.set_title(f'{nameArray[i]} Channel Histogram') 
        fig.tight_layout()
        figureArray.append(fig)
        axArray.append(ax)

    return figureArray, axArray, meanArray, deviationArray
