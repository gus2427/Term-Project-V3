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

class NoiseFrame(customtkinter.CTkFrame):
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

        # Cosntants
        ROI = np.copy(self.app.ROI)

        # Spacial
        self.kernelSize = 3
        self.originalBitmatrix=np.copy(self.app.bitMatrix[ROI[1]:ROI[3],ROI[0]:ROI[2]])
        self.spatialDictionary = {
            "Arithmetic": self.ArithmeticMeanFilter, 
            "Geometric": self.GeometricMeanFilter, 
            "Harmonic": self.HarmonicMeanFilter,
            "Contra Harmonic": self.ContraharmonicMeanFilter,
            "Max": self.maxFilter,
            "Min": self.minFilter,
            "Midpoint": self.midpointFilter, 
            "Median": self.medianFilter, 
            "Alpha Trimmed Mean": self.alphaTrimmedMeanFilter,
            "Adaptive Local Noise": self.adaptiveLocalNoiseReduction,
        }

        self.option = None
        self.Q = 0
        self.Alpha = 1

        # Frequency
        self.FshiftImage = np.fft.fftshift(np.fft.fft2(self.originalBitmatrix))
        self.GshitImage = None
        self.Transferfunction = None
        self.D0 = 10
        self.order = 5
        self.Q_freq = 1
        self.uk_offset = 0
        self.vk_offset = 0
        self.uk = 0
        self.vk = 0
        self.updateDictionary()

        self.CanvasArray=[]
        self.Axs=[]

        for i in range(4):
            self.CanvasArray.append(FigureCanvasTkAgg(plt.figure(figsize=(4, 4),tight_layout=True),master=self.GraphFrame))
            self.Axs.append(plt.axes())

        # region Spacial
        self.SpacialFrame = customtkinter.CTkFrame(self.ActionFrameExtended,border_width=5)
        self.SpacialFrame.rowconfigure((0,1,2,3,4,5,6,7,8,9),weight=0)
        self.SpacialFrame.columnconfigure((0),weight=1)

        # Widgets
        self.Size_SliderLabel=customtkinter.CTkLabel(self.SpacialFrame,text=f"Kermel Size = {self.kernelSize}")
        self.Size_SliderLabel.grid(row=0,column=0,padx=10, pady=20)
        self.Size_Slider = customtkinter.CTkSlider(self.SpacialFrame, from_=1, to=45, number_of_steps=44, command= self.kernalSizeEvent)
        self.Size_Slider.grid(row=1,column=0,padx=10, pady=20)
        self.Size_Slider.set(self.kernelSize)

        self.Q_SliderLabel=customtkinter.CTkLabel(self.SpacialFrame,text=f"Q = {self.Q}")
        self.Q_SliderLabel.grid(row=0,column=1,padx=10, pady=20)
        self.Q_Slider = customtkinter.CTkSlider(self.SpacialFrame, from_=-20, to=20, number_of_steps=40, command=self.QEvent)
        self.Q_Slider.grid(row=1,column=1,padx=10, pady=20)
        self.Q_Slider.set(self.Q)

        self.Alpha_SliderLabel=customtkinter.CTkLabel(self.SpacialFrame,text=f"Alpha = {self.Alpha}")
        self.Alpha_SliderLabel.grid(row=2,column=0,columnspan=2,padx=10, pady=20)
        self.Alpha_Slider = customtkinter.CTkSlider(self.SpacialFrame, from_=1, to=20, number_of_steps=19,command=self.AlphaSizeEvent)
        self.Alpha_Slider.grid(row=3,column=0,columnspan=2,padx=10, pady=20)
        self.Alpha_Slider.set(self.Alpha)
        
        self.MeanOptionsLabel=customtkinter.CTkLabel(self.SpacialFrame,text="Mean Filters")
        self.MeanOptionsLabel.grid(row=4,column=0,columnspan=2,padx=10, pady=20)
        self.MeanOptions= customtkinter.CTkOptionMenu(self.SpacialFrame,values=["Arithmetic", "Geometric", "Harmonic", "Contra Harmonic"], width = 210, dynamic_resizing=False,command=self.SpatialMenus)
        self.MeanOptions.grid(row=5,column=0,columnspan=2,padx=10, pady=20)

        self.StatisticLabel=customtkinter.CTkLabel(self.SpacialFrame,text="Statistic Filters")
        self.StatisticLabel.grid(row=6,column=0,columnspan=2,padx=10, pady=20)
        self.StatisticOptions= customtkinter.CTkOptionMenu(self.SpacialFrame,values=["Max", "Min", "Midpoint", "Median", "Alpha Trimmed Mean"], width = 210, dynamic_resizing=False,command=self.SpatialMenus)
        self.StatisticOptions.grid(row=7,column=0,columnspan=2,padx=10, pady=20)

        self.AdaptiveLabel=customtkinter.CTkLabel(self.SpacialFrame,text="Adaptive Filters")
        self.AdaptiveLabel.grid(row=8,column=0,columnspan=2,padx=10, pady=20)
        self.AdaptiveOptions= customtkinter.CTkOptionMenu(self.SpacialFrame,values=["Adaptive Local Noise"], width = 210, dynamic_resizing=False,command=self.SpatialMenus)
        self.AdaptiveOptions.grid(row=9,column=0,columnspan=2,padx=10, pady=20)

        # endregion

        # region Frequency
        self.FrequencyFrame = customtkinter.CTkFrame(self.ActionFrameExtended,border_width=5)
        self.FrequencyFrame.rowconfigure((0,1,2,3,4,5,6,7,8),weight=0)
        self.FrequencyFrame.columnconfigure((0,1),weight=1)

        self.D0_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"Cutoff frequency = {self.D0}")
        self.D0_SliderLabel.grid(row=0,column=0,padx=10, pady=20)
        self.D0_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=1, to=int(np.power(np.max(self.originalBitmatrix.shape),.75)), number_of_steps=499, command=self.D0_SliderEvent)
        self.D0_Slider.grid(row=1,column=0,padx=10, pady=20)
        self.D0_Slider.set(self.D0)

        self.Order_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"Order = {self.order}")
        self.Order_SliderLabel.grid(row=0,column=1,padx=10, pady=20)
        self.Order_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=0, to=10, number_of_steps=10, command=self.Order_SliderEvent)
        self.Order_Slider.grid(row=1,column=1,padx=10, pady=20)
        self.Order_Slider.set(self.order)

        self.uk_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"uk = {self.uk}")
        self.uk_SliderLabel.grid(row=2,column=0,padx=10, pady=20)
        self.uk_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=-100, to=100, number_of_steps=200, command=self.uk_SliderEvent)
        self.uk_Slider.grid(row=3,column=0,padx=10, pady=20)
        self.uk_Slider.set(self.uk)

        self.vk_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"vk = {self.vk}")
        self.vk_SliderLabel.grid(row=2,column=1, padx=10, pady=20)
        self.vk_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=-100, to=100, number_of_steps=200, command=self.vk_SliderEvent)
        self.vk_Slider.grid(row=3,column=1,padx=10, pady=20)
        self.vk_Slider.set(self.vk)

        self.uk_offset_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"uk offset = {self.uk_offset}")
        self.uk_offset_SliderLabel.grid(row=4,column=0, padx=10, pady=20)
        self.uk_offset_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=-100, to=100, number_of_steps=200, command=self.uk_offset_SliderEvent)
        self.uk_offset_Slider.grid(row=5,column=0,padx=10, pady=20)
        self.uk_offset_Slider.set(self.uk_offset)

        self.vk_offset_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"vk offset = {self.vk_offset}")
        self.vk_offset_SliderLabel.grid(row=4,column=1, padx=10, pady=20)
        self.vk_offset_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=-100, to=100, number_of_steps=200, command=self.vk_offset_SliderEvent)
        self.vk_offset_Slider.grid(row=5,column=1,padx=10, pady=20)
        self.vk_offset_Slider.set(self.vk_offset)

        self.Q_freq_SliderLabel=customtkinter.CTkLabel(self.FrequencyFrame,text=f"Q = {self.Q_freq}")
        self.Q_freq_SliderLabel.grid(row=6,column=0,columnspan=2, padx=10, pady=20)
        self.Q_freq_Slider = customtkinter.CTkSlider(self.FrequencyFrame, from_=1, to=10, number_of_steps=9, command=self.Q_freqSilderEvent)
        self.Q_freq_Slider.grid(row=7,column=0,columnspan=2,padx=10, pady=20)
        self.Q_freq_Slider.set(self.Q_freq)

        self.NotchLabel=customtkinter.CTkLabel(self.FrequencyFrame,text="Notch Filters")
        self.NotchLabel.grid(row=8,column=0,padx=10, pady=20)
        self.NotchOptions= customtkinter.CTkOptionMenu(self.FrequencyFrame,values=["Gaussian","Ideal","Butterworth"], width = 210, dynamic_resizing=False,command=self.FrequencyNotchMenus)
        self.NotchOptions.grid(row=9,column=0,padx=10, pady=20)

        self.NotchPassLabel=customtkinter.CTkLabel(self.FrequencyFrame,text="Notch Pass Filters")
        self.NotchPassLabel.grid(row=8,column=1,padx=10, pady=20)
        self.NotchPassOptions= customtkinter.CTkOptionMenu(self.FrequencyFrame,values=["Gaussian","Ideal","Butterworth"], width = 210, dynamic_resizing=False,command=self.FrequencyNotchPassMenus)
        self.NotchPassOptions.grid(row=9,column=1,padx=10, pady=20)

        # endregion

        # General Functions
    def processNewImage(self,Matrix):
        self.app.stitchROIandDisplay(Matrix)


    def updateInputs(self):
        self.originalBitmatrix=np.copy(self.app.bitMatrix[self.app.ROI[1]:self.app.ROI[3],self.app.ROI[0]:self.app.ROI[2]])
        self.FshiftImage = np.fft.fftshift(np.fft.fft2(self.originalBitmatrix))
        
        self.D0_Slider.configure(from_=1, to=int(np.max(self.originalBitmatrix.shape)))

    def resetNoiseInputs(self):
        # Frequency
        self.FshiftImage = np.fft.fftshift(np.fft.fft2(self.originalBitmatrix))
        self.GshitImage = None
        self.Transferfunction = None
        self.D0 = 10
        self.order = 5
        self.uk = 0
        self.vk = 0
        self.Q_freq = 1
        self.updateDictionary()

        #Spacial
        self.kernelSize = 3
        self.option = None
        self.Q = 0
        self.Alpha = 1

        ungrid_all_widgets(self.ActionFrameExtended)

        # region Spacial
    def SpacialDomainEvent(self):
        ungrid_all_widgets(self.GraphFrame)

        self.app.stitchROIandDisplay(self.originalBitmatrix)
            
        self.resetNoiseInputs()

        self.SpacialFrame.grid(row=0,column=0,sticky= 'nesw')

    def kernalSizeEvent(self,value):
        self.kernelSize=int(value)
        self.Size_SliderLabel.configure(text=f"Kermel Size = {self.kernelSize}")

    def QEvent(self,value):
        self.Q=int(value)
        self.Q_SliderLabel.configure(text=f"Q = {self.Q}")

    def AlphaSizeEvent(self,value):
        self.Alpha=int(value)
        self.Alpha_SliderLabel.configure(text=f"Alpha = {self.Alpha}")

    def SpatialMenus(self,value):
        self.option = value

        result = self.spatialDictionary[value](matrix=self.originalBitmatrix,Q=self.Q,alpha=self.Alpha)

        self.processNewImage(result)
        
    # Mean Filters
    def ArithmeticMeanFilter(self,matrix, Q, alpha):
        matrix = matrix.astype(np.float64)
        matrix = np.maximum(matrix, 1e-10)

        paddedMatrix = self.padMatrix(matrix)

        kernel = np.ones((self.kernelSize, self.kernelSize), np.float64) / (self.kernelSize**2)

        result = scipy.signal.convolve2d(paddedMatrix,kernel,'same')

        resultClipped = clipValues(result,self.app.bits)

        while resultClipped.shape != matrix.shape:
            resultClipped = resultClipped[:-1, :-1]

        return resultClipped
        
    def GeometricMeanFilter(self,matrix, Q, alpha):
        matrix = matrix.astype(np.float64)
        matrix = np.maximum(matrix, 1e-10)
        paddedMatrix = self.padMatrix(matrix)

        result = np.power(np.prod(np.lib.stride_tricks.sliding_window_view(paddedMatrix, (self.kernelSize, self.kernelSize)), axis=(-2, -1)), 1/(self.kernelSize**2))

        resultClipped = clipValues(result,self.app.bits)
        
        while resultClipped.shape != matrix.shape:
            resultClipped = resultClipped[:-1, :-1]

        return resultClipped
    
    def padMatrix(self,matrix):
        padding_needed = self.kernelSize // 2

        pad_width = [(padding_needed, padding_needed),
                    (padding_needed, padding_needed)]
        
        return np.pad(matrix, pad_width, mode='symmetric')

    def HarmonicMeanFilter(self,matrix, Q, alpha):
        kernel = np.ones((self.kernelSize, self.kernelSize), np.float64)

        matrix = matrix.astype(np.float64)
        matrix = np.maximum(matrix, 1e-10)
        paddedMatrix = self.padMatrix(matrix)

        result = (self.kernelSize**2) / scipy.signal.convolve2d(np.power(paddedMatrix,-1,dtype=np.float64),kernel,'same')

        resultClipped = clipValues(result,self.app.bits).astype(np.uint8)

        while resultClipped.shape != matrix.shape:
            resultClipped = resultClipped[:-1, :-1]

        return resultClipped

    def ContraharmonicMeanFilter(self,matrix, Q, alpha):
        kernel = np.ones((self.kernelSize, self.kernelSize), np.float64)

        matrix = np.maximum(matrix, 1e-10)

        Num = (scipy.signal.convolve2d(np.power(matrix,(Q+1),dtype=np.float64),kernel,'same'))
        dem = (scipy.signal.convolve2d(np.power(matrix,Q,dtype=np.float64),kernel,'same'))

        dem = np.maximum(dem, 1e-10)

        result = np.divide(Num,dem, out=np.zeros_like(Num))

        resultClipped = clipValues(result,self.app.bits).astype(np.uint8)

        return resultClipped
    
    #Order-Statistic Filters
    def medianFilter(self,matrix, Q, alpha):
        result = scipy.ndimage.median_filter(matrix, size=self.kernelSize)

        return result
        
    def maxFilter(self, matrix, Q, alpha):
        result = scipy.ndimage.maximum_filter(matrix, size=self.kernelSize)

        return result

    def minFilter(self, matrix, Q, alpha):
        result = scipy.ndimage.minimum_filter(matrix, size=self.kernelSize)

        return result
        
    def midpointFilter(self,matrix, Q, alpha):
        result = (self.maxFilter(matrix, Q, alpha).astype(np.float64) + self.minFilter(matrix, Q, alpha).astype(np.float64)) / 2

        return result.astype(np.uint8)
        
    def alphaTrimmedMeanFilter(self,matrix, Q, alpha):

        def alpha_trimmed_mean(values):
            sorted_values = np.sort(values)
            trimmed_values = sorted_values[alpha//2 : -alpha//2]
            return np.mean(trimmed_values)

        filtered_image = scipy.ndimage.generic_filter(matrix, alpha_trimmed_mean, size=self.kernelSize)

        return filtered_image

    # Adaptive Filters
    def adaptiveLocalNoiseReduction(self,matrix, Q, alpha):
        matrix = matrix.astype(np.float64)
             
        C = np.pad(matrix, ((self.kernelSize // 2, self.kernelSize // 2), (self.kernelSize // 2, self.kernelSize // 2)), mode='constant')

        lvar = np.zeros_like(matrix)
        lmean = np.zeros_like(matrix)
        temp = np.zeros_like(matrix)
        NewImg = np.zeros_like(matrix)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                temp = C[i:i+self.kernelSize, j:j+self.kernelSize]
                tmp = temp.flatten()

                lmean[i, j] = np.mean(tmp)
                lvar[i, j] = np.mean(tmp**2) - np.mean(tmp)**2

        nvar = np.sum(lvar) / matrix.size

        lvar = np.maximum(lvar, nvar)

        NewImg = nvar / lvar
        NewImg = NewImg * (matrix - lmean)
        NewImg = matrix - NewImg

        filtered_image=clipValues(NewImg,self.app.bits).astype(np.uint8)

        return filtered_image

        #adaptive_median_filter

    # endregion
    
    # region Frequency domain
    def FrequencyDomainEvent(self):
        ungrid_all_widgets(self.GraphFrame)

        self.app.stitchROIandDisplay(self.originalBitmatrix)
        
        self.resetNoiseInputs()

        self.FrequencyFrame.grid(row=0,column=0,sticky= 'nesw')

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

    def uk_SliderEvent(self,value):
        self.uk = value
        self.uk_SliderLabel.configure(text=f"uk = {self.uk}")

        self.updateDictionary()
        if self.Transferfunction != None:
            self.processTransfer()

    def vk_SliderEvent(self,value):
        self.vk = value
        self.vk_SliderLabel.configure(text=f"yk = {self.vk}")

        self.updateDictionary()
        if self.Transferfunction != None:
            self.processTransfer()

    def uk_offset_SliderEvent(self,value):
        self.uk_offset = value
        self.uk_offset_SliderLabel.configure(text=f"uk offset = {self.uk_offset}")

        self.updateDictionary()
        if self.Transferfunction != None:
            self.processTransfer()

    def vk_offset_SliderEvent(self,value):
        self.vk_offset = value
        self.vk_offset_SliderLabel.configure(text=f"vk offset = {self.vk_offset}")

        self.updateDictionary()
        if self.Transferfunction != None:
            self.processTransfer()

    def Q_freqSilderEvent(self,value):
        self.Q_freq = int(value)
        self.Q_freq_SliderLabel.configure(text=f"Q = {self.Q_freq}")

        self.updateDictionary()
        if self.Transferfunction != None:
            self.processTransfer()

    def FrequencyNotchMenus(self,value):
        self.Transferfunction = value + " Notch"

        self.processTransfer()

    def FrequencyNotchPassMenus(self,value):
        self.Transferfunction = value + " Notch Pass"

        self.processTransfer()

    # Frequency Filter
    def applyFilter(self,H):
        self.GshitImage = self.FshiftImage * H
        self.G = np.fft.ifftshift(self.GshitImage)
        g=np.abs(np.fft.ifft2(self.G))

        g = clipValues(g,self.app.bits)

        return g

    def graphFrequencyDomain(self):
        self.CanvasArray[0].get_tk_widget().grid(row=0,column=0 ,sticky='NEWS')
            
        self.Axs[0].clear()

        self.Axs[0].imshow(np.log1p(np.abs(self.FshiftImage)), cmap='gray')

        self.Axs[0].set_title("Frequency Domain") 

        self.CanvasArray[0].draw()
        
    def graphTransferFunction(self,H):
        self.CanvasArray[1].get_tk_widget().grid(row=0,column=1 ,sticky='NEWS')
            
        self.Axs[1].clear()

        self.Axs[1].imshow(self.TransferDictionary[H], cmap='gray')

        self.Axs[1].set_title(H) 

        self.CanvasArray[1].draw()

    def graphGshiftDomain(self):
        self.CanvasArray[2].get_tk_widget().grid(row=1,column=0 ,sticky='NEWS')
            
        self.Axs[2].clear()

        self.Axs[2].imshow(np.log1p(np.abs(self.GshitImage)), cmap='gray')

        self.Axs[2].set_title('Filtering Result') 

        self.CanvasArray[2].draw()

    def graphGDomain(self):
        self.CanvasArray[3].get_tk_widget().grid(row=1,column=1 ,sticky='NEWS')
            
        self.Axs[3].clear()

        self.Axs[3].imshow(np.log1p(np.abs(self.G)), cmap='gray')

        self.Axs[3].set_title("Inverse Fourier Transform") 

        self.CanvasArray[3].draw()

    def processTransfer(self):
        s = self.applyFilter(self.TransferDictionary[self.Transferfunction])
        self.app.stitchROIandDisplay(s)

        self.graphFrequencyDomain()
        self.graphTransferFunction(self.Transferfunction)
        self.graphGshiftDomain()
        self.graphGDomain()
        self.update_idletasks()
 
    def updateDictionary(self):
        M,N = self.originalBitmatrix.shape[:2]

        H = np.zeros((M,N))

        u, v = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')

        product = np.ones((M,N))

        Dk = np.zeros((self.Q_freq, M, N))
        Dnk = np.zeros((self.Q_freq, M, N))

        for i in range(self.Q_freq):
            tempK = np.sqrt((u - M / 2 - self.uk - self.uk_offset*i) ** 2 + (v - N / 2 - self.vk - self.vk_offset*i) ** 2)
            tempNK = np.sqrt((u - M / 2 + self.uk + self.uk_offset*i) ** 2 + (v - N / 2 + self.vk + self.vk_offset*i) ** 2)

            Dk[i, :, :] = np.maximum(tempK, 1e-10)
            Dnk[i, :, :] = np.maximum(tempNK, 1e-10)


        for i in range(len(Dk)):
            H[(Dk[i] <= self.D0) | (Dnk[i] <= self.D0)] = 0.0
            H[(Dk[i] > self.D0) & (Dnk[i] > self.D0)] = 1.0
            
            product = product * H

        self.IdealNotch = product

        product = np.ones((M,N))

        for i in range(len(Dk)):
            H = 1 - np.exp(-.5*np.power(Dk[i]*Dnk[i]/self.D0**2,2)) 
            product = product * H
 
        self.GaussianNotch = product

        product = np.ones((M,N))

        for i in range(len(Dk)):
            H = 1/(1 + np.power(self.D0**2/(Dk[i]*Dnk[i]),self.order))
            product = product * H

        self.ButterworthNotch = product

        self.GaussianNotchPass = 1-self.GaussianNotch
        self.IdealNotchPass = 1 - self.IdealNotch
        self.ButterworthNotchPass = 1 - self.ButterworthNotch

        self.TransferDictionary = {
                "Gaussian Notch" : self.GaussianNotch,
                "Gaussian Notch Pass" : self.GaussianNotchPass,
                "Ideal Notch" : self.IdealNotch,
                "Ideal Notch Pass" :  self.IdealNotchPass,
                "Butterworth Notch" : self.ButterworthNotch,
                "Butterworth Notch Pass" : self.ButterworthNotchPass
            }
        

    # endregion
def adaptive_median_filter(image, max_filter_size=7):
    height, width = image.shape
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            output[i, j] = get_adaptive_median_value(image, i, j, max_filter_size)

    return output

def get_adaptive_median_value(image, i, j, max_filter_size):
    height, width = image.shape
    window_size = 3

    while window_size <= max_filter_size:
        window = image[max(0, i - window_size // 2):min(height, i + window_size // 2 + 1),
                       max(0, j - window_size // 2):min(width, j + window_size // 2 + 1)]

        median_value = np.median(window)
        min_val = np.min(window)
        max_val = np.max(window)

        if min_val < median_value < max_val:
            pixel_value = image[i, j]

            if min_val < pixel_value < max_val:
                return pixel_value
            else:
                return median_value

        window_size += 2

    return image[i, j]