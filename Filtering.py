import numpy as np
import matplotlib.pyplot as plt
import customtkinter
import scipy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def ungrid_all_widgets(frame):
    for widget in frame.winfo_children():
        widget.grid_forget()

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def clipValues(Values,bits):

    output = np.clip(Values,0,2**bits-1)

    return output

class FiteringFrame(customtkinter.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, border_width=5)

        self.app = app

        self.grid_rowconfigure((0), weight=0)
        self.grid_rowconfigure((1), weight=1)
        self.grid_columnconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=0)

        # Frame
        self.ActionFrame = customtkinter.CTkFrame(self,border_width=5)
        self.ActionFrame.columnconfigure((0,1),weight=1)
        self.ActionFrame.rowconfigure((0),weight=1)
        self.ActionFrame.grid(row=0,column=0, columnspan=2,sticky="nsew")

        self.SpacialButton = customtkinter.CTkButton(self.ActionFrame, text="Spatial Domain", command=self.SpacialDomainEvent)
        self.SpacialButton.grid(row=0,column=0,padx=10, pady=20)
        
        self.FrequencyButton = customtkinter.CTkButton(self.ActionFrame, text="Frequency Domain", command=self.FrequencyDomainEvent)
        self.FrequencyButton.grid(row=0,column=1,padx=10, pady=20)

        self.GraphFrame = customtkinter.CTkFrame(self,border_width=5)
        self.GraphFrame.columnconfigure((0,1),weight=1)
        self.GraphFrame.rowconfigure((0,1),weight=1)
        self.GraphFrame.grid(row=1,column=0,sticky="nsew")

        self.ActionFrameExtended = customtkinter.CTkFrame(self,border_width=5)
        self.ActionFrameExtended.rowconfigure((0),weight=1)
        self.ActionFrameExtended.columnconfigure((0),weight=1)
        self.ActionFrameExtended.grid(row=1,column=1,sticky="nsew")

        # Constants
        self.ROI = np.copy(self.app.ROI)

        #Spacial
        self.kernelSize = 3
        self.std = 1
        self.originalBitmatrix=np.copy(self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]])
        self.seletedKernel = None

        # Frequency
        self.FshiftImage = np.fft.fftshift(np.fft.fft2(self.originalBitmatrix))
        self.GshitImage = None
        self.D0 = 10
        self.order = 5
        self.Transferfunction = None

        # TransferFunctions
        self.updateDictionary()

        self.filterCanvasArray=[]
        self.filter_Axs=[]

        for i in range(5):
            self.filterCanvasArray.append(FigureCanvasTkAgg(plt.figure(figsize=(4, 4),tight_layout=True),master=self.GraphFrame))
            self.filter_Axs.append(plt.axes())

        # Spacial Operators
        self.BoxBlur = (1 / self.kernelSize**2) * np.ones((self.kernelSize,self.kernelSize))

        self.GaussianKernerl = gaussian_kernel(self.kernelSize,self.std)

        self.Laplacian1 = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, -0]])
        
        self.Laplacian2 = np.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]])
        
        self.Laplacian3 = self.Laplacian1 * -1
        
        self.Laplacian4 = self.Laplacian2 * -1       


        self.HorizontalSobelFilter = np.array([[1, 2, 1],
                                            [0, 0, 0],
                                            [-1, -2, -1]])

        self.VerticalSobelFilter = np.array([[1, 0, -1],
                                            [2, 0, -2],
                                            [1, 0, -1]])

        self.LeftDiagonalFilter = np.array([[1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1]])

        self.RightDiagonalFilter = np.array([[-1, -1, 1],
                                            [-1, 1, -1],
                                            [1, -1, -1]])

        self.spacialDict = {
            "Horizontal Sobel" : self.HorizontalSobelFilter,
            "Vertical Sobel" : self.VerticalSobelFilter,
            "Left Diagonal Sobel" : self.LeftDiagonalFilter,
            "Right Diagonal Sobel" : self.RightDiagonalFilter,
            "Laplacian 1" : self.Laplacian1,
            "Laplacian 2" : self.Laplacian2,
            "Laplacian 3" : self.Laplacian3,
            "Laplacian 4" : self.Laplacian4,
            "Box Blur" : self.BoxBlur,
            "Gaussian Blur" : self.GaussianKernerl
        }
        # Main Elements
        # Frequency
        self.FrequencyFrame = customtkinter.CTkFrame(self.ActionFrameExtended,border_width=5)
        self.FrequencyFrame.rowconfigure((0,1,2,3,4),weight=0)
        self.FrequencyFrame.columnconfigure((0,1),weight=1)

        # Widgets
        self.D0_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"Cutoff frequency = {self.D0}")
        self.D0_SliderLabel.grid(row=0,column=0,padx=10, pady=20)
        self.D0_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=1, to=int(np.max(self.originalBitmatrix.shape)), number_of_steps=499, command=self.D0_SliderEvent)
        self.D0_Slider.grid(row=1,column=0,padx=10, pady=20)
        self.D0_Slider.set(10)

        self.Order_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"Order = {self.order}")
        self.Order_SliderLabel.grid(row=0,column=1,padx=10, pady=20)
        self.Order_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=0, to=10, number_of_steps=10, command=self.Order_SliderEvent)
        self.Order_Slider.grid(row=1,column=1,padx=10, pady=20)
        self.Order_Slider.set(5)

        self.SmoothingLabel=customtkinter.CTkLabel(self.FrequencyFrame,text="Smoothing",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.SmoothingLabel.grid(row=2,column=0,padx=10, pady=20)
        self.LPFLabel=customtkinter.CTkLabel(self.FrequencyFrame,text="Low Pass Filter",font=customtkinter.CTkFont(size=17, weight="bold"))
        self.LPFLabel.grid(row=3,column=0,padx=10, pady=20)

        self.SharpeningLabel=customtkinter.CTkLabel(self.FrequencyFrame,text="Sharpening",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.SharpeningLabel.grid(row=2,column=1,padx=10, pady=20)
        self.HPFLabel=customtkinter.CTkLabel(self.FrequencyFrame,text="High Pass Filter",font=customtkinter.CTkFont(size=17, weight="bold"))
        self.HPFLabel.grid(row=3,column=1,padx=10, pady=20)

        self.LPFOptions= customtkinter.CTkOptionMenu(self.FrequencyFrame,values=["Gaussian", "Ideal", "Butterworth"], dynamic_resizing=False,command=self.FrequencyLPFMenus)
        self.LPFOptions.grid(row=4,column=0,padx=10, pady=20)

        self.HPFOptions= customtkinter.CTkOptionMenu(self.FrequencyFrame,values=["Gaussian", "Ideal", "Butterworth"], dynamic_resizing=False,command=self.FrequencyHPFMenus)
        self.HPFOptions.grid(row=4,column=1,padx=10, pady=20)

        self.addFDShapening = customtkinter.CTkButton(self.FrequencyFrame, text="Add to Original", command=self.addShapeningEvent)
        self.addFDShapening.grid(row=5,column=0,columnspan=2,padx=10, pady=20)

        # Spacial
        self.SpacialFrame = customtkinter.CTkFrame(self.ActionFrameExtended,border_width=5)
        self.SpacialFrame.rowconfigure((0,1,2,3,4),weight=0)
        self.SpacialFrame.columnconfigure((0,1),weight=1)

        # Widgets
        self.SmoothingSpacialLabel=customtkinter.CTkLabel(self.SpacialFrame,text="Smoothing",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.SmoothingSpacialLabel.grid(row=0,column=0,padx=10, pady=20)

        self.Size_SliderLabel=customtkinter.CTkLabel(self.SpacialFrame,text=f"Kermel Size = {self.kernelSize}")
        self.Size_SliderLabel.grid(row=1,column=0,padx=10, pady=20)
        self.Size_Slider = customtkinter.CTkSlider(self.SpacialFrame, from_=1, to=45, number_of_steps=44, command=self.kernalSizeEvent)
        self.Size_Slider.grid(row=2,column=0,padx=10, pady=20)
        self.Size_Slider.set(self.kernelSize)

        self.std_SliderLabel=customtkinter.CTkLabel(self.SpacialFrame,text=f"std = {self.std}")
        self.std_SliderLabel.grid(row=3,column=0,padx=10, pady=20)
        self.std_Slider = customtkinter.CTkSlider(self.SpacialFrame, from_=1, to=25, number_of_steps=24, command=self.stdEvent)
        self.std_Slider.grid(row=4,column=0,padx=10, pady=20)
        self.std_Slider.set(self.std)

        self.SmoothingKernelsOptions= customtkinter.CTkOptionMenu(self.SpacialFrame,values=["Box Blur", "Gaussian Blur"], dynamic_resizing=False,command=self.SpatialMenus)
        self.SmoothingKernelsOptions.grid(row=5,column=0,padx=10, pady=20)

        self.SharpeningSpacialLabel=customtkinter.CTkLabel(self.SpacialFrame,text="Sharpening",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.SharpeningSpacialLabel.grid(row=0,column=1,padx=10, pady=20)

        self.SharpeningKernelsOptions= customtkinter.CTkOptionMenu(self.SpacialFrame,values=["Laplacian 1", "Laplacian 2", "Laplacian 3","Laplacian 4"], dynamic_resizing=False,command=self.SpatialMenus)
        self.SharpeningKernelsOptions.grid(row=1,column=1,padx=10, pady=20)

        self.miscSpacialLabel=customtkinter.CTkLabel(self.SpacialFrame,text="Edges",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.miscSpacialLabel.grid(row=2,column=1,padx=10, pady=20)

        self.EdgeKernelsOptions = customtkinter.CTkOptionMenu(self.SpacialFrame,values=["Horizontal Sobel", "Vertical Sobel", "Left Diagonal Sobel","Right Diagonal Sobel"],width=175, dynamic_resizing=False,command=self.SpatialMenus)
        self.EdgeKernelsOptions.grid(row=3,column=1,padx=10, pady=20)
        
        self.addShapening = customtkinter.CTkButton(self.SpacialFrame, text="Add to Original", command=self.addShapeningEvent)
        self.addShapening.grid(row=4,column=1,padx=10, pady=20)


    # Reset Functions
    def resetFilterInputs(self):
        # Frequency
        self.FshiftImage = np.fft.fftshift(np.fft.fft2(self.originalBitmatrix))
        self.GshitImage = None
        self.D0 = 10
        self.order = 5
        self.Transferfunction = None

        #Spacial
        self.kernelSize = 3
        self.std = 1
        self.seletedKernel = None

        self.updateDictionary()
        ungrid_all_widgets(self.ActionFrameExtended)

    def updateInputs(self):
        self.ROI = self.app.ROI
        self.originalBitmatrix=np.copy(self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]])
        self.FshiftImage = np.fft.fftshift(np.fft.fft2(self.originalBitmatrix))
        
        self.D0_Slider.configure(from_=1, to=int(np.max(self.originalBitmatrix.shape)))

    def updateDictionary(self):
        M,N = self.originalBitmatrix.shape[:2]

        u, v = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        self.GaussianLPF = np.exp(-D ** 2 / (2 * self.D0 ** 2))
            
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        self.IdealLPF = np.where(D,D<=self.D0,1)

        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        self.ButterworthLPF = 1/(1+ (D/self.D0)**(2*self.order))

        self.GaussianHPF = 1-self.GaussianLPF
        self.IdealHPF = 1 - self.IdealLPF
        self.ButterworthHPF = 1 - self.ButterworthLPF

        self.TransferDictionary = {
                "Gaussian Low Pass" : self.GaussianLPF,
                "Gaussian High Pass" : self.GaussianHPF,
                "Ideal Low Pass" : self.IdealLPF,
                "Ideal High Pass" :  self.IdealHPF,
                "Butterworth Low Pass" : self.ButterworthLPF,
                "Butterworth High Pass" : self.ButterworthHPF
            }
        
    def FrequencyDomainEvent(self):
        ungrid_all_widgets(self.GraphFrame)

        s = self.originalBitmatrix
        self.app.stitchROIandDisplay(s)
        
        self.resetFilterInputs()
        self.FrequencyFrame.grid(row=0,column=0,sticky= 'nesw')

    def applyFilter(self,H):
        self.GshitImage = self.FshiftImage * H
        self.G = np.fft.ifftshift(self.GshitImage)
        g=np.abs(np.fft.ifft2(self.G))

        g = clipValues(g,self.app.bits)
        
        return g

    def graphFrequencyDomain(self):
        self.filterCanvasArray[0].get_tk_widget().grid(row=0,column=0 ,sticky='NEWS')
            
        self.filter_Axs[0].clear()

        self.filter_Axs[0].imshow(np.log1p(np.abs(self.FshiftImage)), cmap='gray')

        self.filter_Axs[0].set_title("Frequency Domain") 

        self.filterCanvasArray[0].draw()
        
    def graphTransferFunction(self,H):
        self.filterCanvasArray[1].get_tk_widget().grid(row=0,column=1 ,sticky='NEWS')
            
        self.filter_Axs[1].clear()

        self.filter_Axs[1].imshow(self.TransferDictionary[H], cmap='gray')

        self.filter_Axs[1].set_title(H) 

        self.filterCanvasArray[1].draw()

    def graphGshiftDomain(self):
        self.filterCanvasArray[2].get_tk_widget().grid(row=1,column=0 ,sticky='NEWS')
            
        self.filter_Axs[2].clear()

        self.filter_Axs[2].imshow(np.log1p(np.abs(self.GshitImage)), cmap='gray')

        self.filter_Axs[2].set_title('Filtering Result') 

        self.filterCanvasArray[2].draw()

    def graphGDomain(self):
        self.filterCanvasArray[3].get_tk_widget().grid(row=1,column=1 ,sticky='NEWS')
            
        self.filter_Axs[3].clear()

        self.filter_Axs[3].imshow(np.log1p(np.abs(self.G)), cmap='gray')

        self.filter_Axs[3].set_title("Inverse Fourier Transform") 

        self.filterCanvasArray[3].draw()

    def processTransfer(self):
        s = self.applyFilter(self.TransferDictionary[self.Transferfunction])
        self.app.stitchROIandDisplay(s)

        self.graphFrequencyDomain()
        self.graphTransferFunction(self.Transferfunction)
        self.graphGshiftDomain()
        self.graphGDomain()
        self.update_idletasks()

    def D0_SliderEvent(self,value):
        self.D0 = round(value,2)
        self.D0_SliderLabel.configure(text=f"Cutoff frequency = {self.D0}")

        self.updateDictionary()
        if self.Transferfunction != None:
            self.processTransfer()

    def Order_SliderEvent(self,value):
        self.order = value
        self.Order_SliderLabel.configure(text=f"Order = {self.order}")

        self.updateDictionary()
        if self.Transferfunction != None:
            self.processTransfer()

    def FrequencyLPFMenus(self,value):
        self.Transferfunction = value + " Low Pass"

        self.processTransfer()

    def FrequencyHPFMenus(self,value):
        self.Transferfunction = value + " High Pass"

        self.processTransfer()


    # Spaical
    def SpacialDomainEvent(self):
        ungrid_all_widgets(self.GraphFrame)

        s = self.originalBitmatrix
        self.app.stitchROIandDisplay(s)
        
        self.resetFilterInputs()

        self.SpacialFrame.grid(row=0,column=0,sticky= 'nesw')

    def applySacialFilter(self,kernel):
        result = scipy.signal.convolve2d(self.originalBitmatrix,kernel,'same')
        result = clipValues(result,self.app.bits).astype(np.uint8)
        return result
    
    def updateSpacialDictionary(self):
        self.BoxBlur = (1 / self.kernelSize**2) * np.ones((self.kernelSize,self.kernelSize))

        self.GaussianKernerl = gaussian_kernel(self.kernelSize,self.std)

        self.spacialDict["Box Blur"] = self.BoxBlur
        self.spacialDict["Gaussian Blur"] = self.GaussianKernerl

        if self.seletedKernel == "Box Blur" or self.seletedKernel == "Gaussian Blur":
            self.processSpacial()

    def graphKernel(self,kernel):
        self.filterCanvasArray[0].get_tk_widget().grid(row=0,column=0 ,sticky='NEWS')
            
        self.filter_Axs[0].clear()

        self.filter_Axs[0].imshow(self.spacialDict[kernel], cmap='gray', interpolation='none')

        self.filter_Axs[0].set_title(f"{kernel} kernel") 

        for i in range(self.spacialDict[kernel].shape[0]):
            for j in range(self.spacialDict[kernel].shape[1]):
                text = f'{self.spacialDict[kernel][i, j]:.2f}'
                self.filter_Axs[0].text(j, i, text, ha='center', va='center', color='r', fontsize=10)

        self.filterCanvasArray[0].draw()

    def processSpacial(self):
        s = self.applySacialFilter(self.spacialDict[self.seletedKernel])
        self.app.stitchROIandDisplay(s)

        self.graphKernel(self.seletedKernel)

    def kernalSizeEvent(self,value):
        self.kernelSize=int(value)
        self.Size_SliderLabel.configure(text=f"Kermel Size = {self.kernelSize}")
        self.updateSpacialDictionary()

    def stdEvent(self,value):
        self.std=value
        self.std_SliderLabel.configure(text=f"Std = {self.std}")

        self.updateSpacialDictionary()

    def SpatialMenus(self,value):
        self.seletedKernel = value
        self.processSpacial()

    def addShapeningEvent(self):
        s = self.originalBitmatrix.astype(np.float64) + self.app.bitMatrix[self.ROI[1]:self.ROI[3],self.ROI[0]:self.ROI[2]].astype(np.float64)
        s=clipValues(s,self.app.bits).astype(np.uint8)
        self.app.stitchROIandDisplay(s)

