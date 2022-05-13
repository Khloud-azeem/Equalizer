from PyQt5 import QtWidgets, QtCore, uic, QtGui, QtPrintSupport
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *   
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from os import path
import numpy as np
import sys
import os
import math
import pyqtgraph.exporters
from scipy import signal
from fpdf import FPDF
import sounddevice as sd
from scipy.fft import fft, fftfreq, rfft, rfftfreq , irfft
import librosa
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write

MAIN_WINDOW,_=loadUiType(path.join(path.dirname(__file__),"equalizer.ui"))
MAIN_WINDOW2,_=loadUiType(path.join(path.dirname(__file__),"main.ui"))

class mainwind(QMainWindow,MAIN_WINDOW2):
    def __init__(self):
        super(mainwind,self).__init__()
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.create_MenuBar()
    def create_MenuBar(self):
        menuBar=self.menuBar()
        self.setMenuBar(menuBar)
        file_menu= menuBar.addMenu("Equilazer")
        equalizer_action= QAction('Equalizer',self)
        equalizer_action.triggered.connect(self.newwindow)
        file_menu.addAction(equalizer_action)
    def newwindow(self):
        new= MainApp()
        new.show()
        
class MainApp(QMainWindow,MAIN_WINDOW):

    ##default color pallete
    RGB_Pallete1 = (0, 182, 188, 255)
    RGB_Pallete2 = (246, 111, 0, 255)
    RGB_Pallete3 = (75, 0, 113, 255)
    
    ## min , max pixel intensity
    min_list = [0.5,0.5,0.5]
    max_list = [1.0,1.0,1.0]
    
    vSliders = []
    
    labels = []
    
    gainArrays = []
    
    beforWidget_list = []
    afterWidger_list = []
    spectroWidget_list = []
    
    comboBox_list = []
    
    samples_list = [None,None,None]
    sampling_rate_list = [None,None,None]
    T_list = [None,None,None]
    dataLength_list = [None,None,None]
    
    graph_rangeMin = [0,0,0]
    graph_rangeMax = [1000,1000,1000]
    
    current_samples = [[],[],[]]
    
    ##for graph speed
    step = 5
    def __init__(self):
        super(MainApp,self).__init__()
        QMainWindow.__init__(self)
        self.setupUi(self) 
        ## lists of everything
        self.comboBox_list = [self.comboBox , self.comboBox_2 , self.comboBox_3]
        
        self.beforeWidget_list = [self.beforeWidget , self.beforeWidget_2 , self.beforeWidget_3]
        
        self.afterWidget_list = [self.afterWidget , self.afterWidget_2 , self.afterWidget_3]
        
        self.spectroWidget_list = [self.spectroWidget , self.spectroWidget_2 , self.spectroWidget_3]
        
        self.spectroSlider1_list = [self.spectroSlider1, self.spectroSlider1_2, self.spectroSlider1_3]
        self.spectroSlider2_list = [self.spectroSlider2, self.spectroSlider2_2, self.spectroSlider2_3]
        
        self.vSliders = [[self.vSlider1,self.vSlider2,self.vSlider3,self.vSlider4,self.vSlider5,self.vSlider6,self.vSlider7,self.vSlider8,self.vSlider9,self.vSlider10]
                        ,[self.vSlider1_2,self.vSlider2_2,self.vSlider3_2,self.vSlider4_2,self.vSlider5_2,self.vSlider6_2,self.vSlider7_2,self.vSlider8_2,self.vSlider9_2,self.vSlider10_2]
                        ,[self.vSlider1_3,self.vSlider2_3,self.vSlider3_3,self.vSlider4_3,self.vSlider5_3,self.vSlider6_3,self.vSlider7_3,self.vSlider8_3,self.vSlider9_3,self.vSlider10_3]]
        
        self.labels = [self.label,self.label_2,self.label_3,self.label_4,self.label_5,self.label_6,self.label_7,self.label_8,self.label_9,self.label_10]
        
        self.connect_func()

######################################## Connect function #########################################

    def connect_func(self):
        ##connectng each button by its function
        
        self.playBtn.triggered.connect(self.start)
        self.OpenSignalBtn.triggered.connect(self.BrowseSignal)
        self.saveBtn.triggered.connect(self.export)
        self.pauseBtn.triggered.connect(self.pause)
        self.stopBtn.triggered.connect(self.stop)
        self.zoomInBtn.triggered.connect(lambda: self.zoom_in_out(0.5))
        self.zoomOutBtn.triggered.connect(lambda: self.zoom_in_out(2))
        self.leftBtn.triggered.connect(lambda: self.move_right_left(-100))
        self.rightBtn.triggered.connect(lambda: self.move_right_left(100))
        self.speedBtn.triggered.connect(self.speedUp)
        self.slowBtn.triggered.connect(self.speedDown)
        self.spectroShow.clicked.connect(self.show_hide)
        self.spectroSliderLabel1.setText("0.5")
        self.spectroSliderLabel2.setText("1")
        self.speedLabel.setText("1x")
        
        for i in range(3):
            self.spectroSlider1_list[i].setMinimum(0)
            self.spectroSlider2_list[i].setMinimum(5)
            self.spectroSlider1_list[i].setMaximum(5)
            self.spectroSlider2_list[i].setMaximum(10)
            self.spectroSlider2_list[i].setValue(10)
            self.spectroSlider1_list[i].setValue(5)
            self.spectroSlider1_list[i].valueChanged.connect(lambda: self.update_spectro())
            self.spectroSlider2_list[i].valueChanged.connect(lambda: self.update_spectro())
            
            self.comboBox_list[i].currentIndexChanged.connect(self.check_pallete)
            
            temp = self.vSliders[i]
            for x in range(10):
                temp[x].valueChanged.connect(self.slidersGains) 


############################################browse function###########################################

    def BrowseSignal(self):
        ##browse audio file
        global name
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","WAV Files (*.wav)")
        name = fileName.split("/")[-1]
        
        ##read the audio file and store it in samples and sampling rate lists
        self.samples_list[self.tabWidget.currentIndex()], self.sampling_rate_list[self.tabWidget.currentIndex()] = librosa.load(fileName, sr=None, mono=True, offset=0.0, duration=None)
        samples = self.samples_list[self.tabWidget.currentIndex()]
        self.current_samples[self.tabWidget.currentIndex()] = samples
        sampling_rate = self.sampling_rate_list[self.tabWidget.currentIndex()]
        
        ## store length and time on their lists
        l=len(samples)
        self.T_list[self.tabWidget.currentIndex()] = int(l / sampling_rate)
        T = self.T_list[self.tabWidget.currentIndex()]
        self.dataLength_list[self.tabWidget.currentIndex()] = l
        
        ##plot spectro and audio before equalizing
        self.plot_spectro(samples[:T*sampling_rate],sampling_rate)
        self.plotBefore(samples,sampling_rate,l)
        ##save a pic of spectro before equalizing
        exporter = pg.exporters.ImageExporter(self.spectroWidget_list[self.tabWidget.currentIndex()].plotItem)
        exporter.export('spectroBefore'+str(self.tabWidget.currentIndex()+1)+'.png')

################################################### plot t-domain before equalizing ###############################

    def plotBefore(self,file,sampling_rate,length):
        ##plot before equalizing
        self.stop()
        
        ## fourier part 
        global yf, xf , phase , magnitude
        n=length
        T=1.0/sampling_rate
        yf = rfft(file)
        magnitude = np.abs(yf)
        xf = rfftfreq(n,T)
        phase = np.angle(yf)
        
        self.beforeWidget_list[self.tabWidget.currentIndex()].clear()
        self.beforeWidget_list[self.tabWidget.currentIndex()].plot(file[0:sampling_rate], pen="r")
        self.beforeWidget_list[self.tabWidget.currentIndex()].setLabel('bottom', "Time", units='s')
        self.beforeWidget_list[self.tabWidget.currentIndex()].setLabel('left', "Amplitude")
        
        self.beforeWidget_list[self.tabWidget.currentIndex()].setLimits(xMin = 0, xMax=xf[-1])
        self.beforeWidget_list[self.tabWidget.currentIndex()].plotItem.setTitle(name)
        self.beforeWidget_list[self.tabWidget.currentIndex()].enableAutoRange(axis='y')

################################# plot t-domain after equalizing #######################

    def plotAfter(self,file,sampling_rate,length):
        ##plot after equalizing
        n=length
        T=1/sampling_rate
        xff = rfftfreq(n,T)
        
        self.afterWidget_list[self.tabWidget.currentIndex()].clear()
        
        self.afterWidget_list[self.tabWidget.currentIndex()].plot(file[0:sampling_rate], pen="b")
        self.afterWidget_list[self.tabWidget.currentIndex()].setLabel('bottom', "Time", units='s')
        self.afterWidget_list[self.tabWidget.currentIndex()].setLabel('left', "Amplitude")
        
        self.afterWidget_list[self.tabWidget.currentIndex()].setLimits(xMin = 0, xMax=xff[-1])
        self.afterWidget_list[self.tabWidget.currentIndex()].plotItem.setTitle(name+ " Equalized")
        self.afterWidget_list[self.tabWidget.currentIndex()].enableAutoRange(axis = "y")
        sd.play(file, sampling_rate)


    def play_audio(self):
        sd.play(self.samples_list[self.tabWidget.currentIndex()], self.sampling_rate_list[self.tabWidget.currentIndex()])

#################################### Spectro function ############################################

    def plot_spectro(self, file , fs):
        #### self.spectroWidget is the plot widget u can change it
        self.spectroWidget_list[self.tabWidget.currentIndex()].clear()
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        
        # the function that plot spectrogram of the selected signal
        f, t, Sxx = signal.spectrogram(file,fs)
        flen = len(f) - 1
        # Item for displaying image data
        img = pg.ImageItem()
        self.spectroWidget_list[self.tabWidget.currentIndex()].addItem(img)
        # Add a histogram with which to control the gradient of the image
        hist = pg.HistogramLUTItem()
        # Link the histogram to the image
        hist.setImageItem(img)
        # Fit the min and max levels of the histogram to the data available
        hist.setLevels(min = np.min(Sxx) , max = np.max(Sxx))
        # This gradient is roughly comparable to the gradient used by Matplotlib
        # You can adjust it and then save it using hist.gradient.saveState()
        min_ = self.min_list[self.tabWidget.currentIndex()]
        max_ = self.max_list[self.tabWidget.currentIndex()]
        hist.gradient.restoreState(
        {'mode': 'rgb','ticks': [(min_, self.RGB_Pallete1)
                                ,(max_, self.RGB_Pallete2)
                                ,(0.0, self.RGB_Pallete3)]})
        
        # Sxx contains the amplitude for each pixel
        img.setImage(Sxx)
        # Scale the X and Y Axis to time and frequency (standard is pixels)
        img.scale(t[-1]/np.size(Sxx, axis=1),f[-1]/np.size(Sxx, axis=0))
        # Limit panning/zooming
        self.spectroWidget_list[self.tabWidget.currentIndex()].setLimits(xMin=t[0], xMax=t[-1], yMin=f[0], yMax=f[-1])
        
        self.spectroWidget_list[self.tabWidget.currentIndex()].setLabel('bottom', "Time", units='s')
        self.spectroWidget_list[self.tabWidget.currentIndex()].setLabel('left', "Frequency", units='Hz')
        self.spectroWidget_list[self.tabWidget.currentIndex()].plotItem.setTitle("Spectrogram")
    
##################################### min, max spectro sliders ############################################    

    def update_spectro(self):
        samples = self.current_samples[self.tabWidget.currentIndex()]
        sampling_rate = self.sampling_rate_list[self.tabWidget.currentIndex()]
        T = self.T_list[self.tabWidget.currentIndex()]
        min_slider = self.spectroSlider1_list[self.tabWidget.currentIndex()]
        max_slider = self.spectroSlider2_list[self.tabWidget.currentIndex()]
        
        self.min_list[self.tabWidget.currentIndex()] = (self.spectroSlider1_list[self.tabWidget.currentIndex()].value())/10
        self.max_list[self.tabWidget.currentIndex()] = (self.spectroSlider2_list[self.tabWidget.currentIndex()].value())/10
        self.plot_spectro(samples[:T*sampling_rate],sampling_rate)
    
    ########################################## sliders function ################################
    
    def slidersGains(self):
        ##check sliders gains
        
        gainArray1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gainArray2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gainArray3 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        self.gainArrays = [gainArray1,gainArray2,gainArray3]
        
        tempgainArray = []
        temp = self.vSliders[self.tabWidget.currentIndex()]
        
        for i in range(10):
            tempgainArray.append(temp[i].value())
            
        self.gainArrays[self.tabWidget.currentIndex()] = tempgainArray
        print(self.gainArrays[self.tabWidget.currentIndex()])
        
        self.processing(self.gainArrays[self.tabWidget.currentIndex()])


################################## color pallete fucntion #########################################

    def check_pallete(self):
        ##to change color pallete of spectro
        
        samples = self.current_samples[self.tabWidget.currentIndex()]
        sampling_rate = self.sampling_rate_list[self.tabWidget.currentIndex()]
        T = self.T_list[self.tabWidget.currentIndex()]
        
        ##setting new values of rgb for each combobox option
        if self.comboBox_list[self.tabWidget.currentIndex()].currentText() == "Default":
            self.RGB_Pallete1 = (0, 182, 188, 255)
            self.RGB_Pallete2 = (246, 111, 0, 255)
            self.RGB_Pallete3 = (75, 0, 113, 255)
            self.plot_spectro(samples[:T*sampling_rate],sampling_rate)
            
        if self.comboBox_list[self.tabWidget.currentIndex()].currentText() == "Palette1":
            self.RGB_Pallete1 = (108, 79, 60, 255)
            self.RGB_Pallete2 = (100, 83, 148, 255)
            self.RGB_Pallete3 = (0, 166, 140, 255)
            self.plot_spectro(samples[:T*sampling_rate],sampling_rate)
            
        if self.comboBox_list[self.tabWidget.currentIndex()].currentText() == "Palette2":
            self.RGB_Pallete1 = (0, 255, 0, 255)
            self.RGB_Pallete2 = (255, 0, 0, 255)
            self.RGB_Pallete3 = (0, 0, 255, 255)
            self.plot_spectro(samples[:T*sampling_rate],sampling_rate)
            
        if self.comboBox_list[self.tabWidget.currentIndex()].currentText() == "Palette3":
            self.RGB_Pallete1 = (219, 178, 209, 255)
            self.RGB_Pallete2 = (147, 71, 66, 255)
            self.RGB_Pallete3 = (108, 160, 220, 255)
            self.plot_spectro(samples[:T*sampling_rate],sampling_rate)
            
        if self.comboBox_list[self.tabWidget.currentIndex()].currentText() == "Palette4":
            self.RGB_Pallete1 = (236, 219, 83, 255)
            self.RGB_Pallete2 = (227, 65, 50, 255)
            self.RGB_Pallete3 = (219, 178, 209, 255)
            self.plot_spectro(samples[:T*sampling_rate],sampling_rate)

######################################### creating bands and multiply it by sliders gain function #########################

    def processing(self,gain):
        ## multiplay the gain of sliders to the audio list
        self.update_labels()
        
        sampling_rate = self.sampling_rate_list[self.tabWidget.currentIndex()]
        T = self.T_list[self.tabWidget.currentIndex()]
        #for dividing 10 ranges 
        eq_range = len(magnitude)//10
        band=[[],[],[],[],[],[],[],[],[],[]]
        temp = magnitude
        
        # multiply bands by its gains and create 2-d array
        for j in range(10):
            for i in range(eq_range):
                band[j].append(temp[i+eq_range*j]*gain[j])
        band[9].append(temp[-1]*gain[9])
        
        # create 1-d array from previous array
        samples_after = []
        for i in range(10):
            for x in range(eq_range):
                samples_after.append(band[i][x])
        samples_after.append(band[9][-1])
        
        l_after = len(samples_after)
        
        # getting phase 
        after_ = []
        for x in range(l_after):
            after_.append(samples_after[x]*(math.cos(phase[x]) +1j*math.sin(phase[x])))   

        # fft inverse
        samplesinv = irfft(after_)
        
        self.current_samples[self.tabWidget.currentIndex()] = samplesinv
        
        self.plotAfter(samplesinv,sampling_rate,l_after)
        self.plot_spectro(samplesinv[:T*sampling_rate],sampling_rate)

    def update_labels(self):
        sampling_rate = self.sampling_rate_list[self.tabWidget.currentIndex()]
        freq = np.arange(sampling_rate * 0.5)
        size = len(freq) / 10
        
        ## setting the ranges of hz that changing for each slider in labels
        self.labels[0].setText(str(freq[21])+"-"+str(freq[int(size)]))
        for i in range(8):
            self.labels[i+1].setText(str(freq[(i+1)*int(size)])+"-"+str(freq[(i+2)*int(size)]))
        self.labels[9].setText(str(freq[9 * int(size)])+"-"+str(freq[-1]))

####################################### Show/Hide spectrogram ##################################

    def show_hide(self ) :
        
        if (self.spectroShow.isChecked()) :
            for i in range(3):
                self.spectroWidget_list[i].hide()
                self.comboBox_list[i].hide()
                self.spectroSlider1_list[i].hide()
                self.spectroSlider2_list[i].hide()
                self.spectroSliderLabel1.hide()
                self.spectroSliderLabel2.hide()
        else :
            for i in range(3):
                self.spectroWidget_list[i].show()
                self.comboBox_list[i].show()
                self.spectroSlider1_list[i].show()
                self.spectroSlider2_list[i].show()
                self.spectroSliderLabel1.show()
                self.spectroSliderLabel2.show()

        ################################ functions from task 1 #####################################
    def speedUp(self):
        if self.step <= 50 :
            self.step +=5
            self.speedLabel.setText(str(self.step/5)+"x")
        else:
            pass
    def speedDown(self):
        if self.step >= 0:
            self.step -=5
            self.speedLabel.setText(str(self.step/5)+"x")
        else:
            pass

    def start(self):
        # the function that makes the graph starts to move
        self.isPaused = False
        self.isStoped = False
        data_length = self.dataLength_list[self.tabWidget.currentIndex()]
        self.play_audio()
        for x in range(0, data_length, self.step):
            #increasing the x-axis range by x
            self.beforeWidget_list[self.tabWidget.currentIndex()].setXRange(self.graph_rangeMin[self.tabWidget.currentIndex()] + x, self.graph_rangeMax[self.tabWidget.currentIndex()] + x)
            self.afterWidget_list[self.tabWidget.currentIndex()].setXRange(self.graph_rangeMin[self.tabWidget.currentIndex()] + x, self.graph_rangeMax[self.tabWidget.currentIndex()] + x)
            QtWidgets.QApplication.processEvents()
            
            if self.isPaused == True:
                #saving the new x-axis ranges
                self.graph_rangeMin[self.tabWidget.currentIndex()] += x
                self.graph_rangeMax[self.tabWidget.currentIndex()] += x
                break
            if self.isStoped == True:
                break

    def pause(self):
        self.isPaused = True
    
    def stop(self):
        #the function that stops the graph
        
        self.isStoped = True
        # reset the graph ranges
        self.beforeWidget_list[self.tabWidget.currentIndex()].enableAutoRange(axis = "x")
        self.afterWidget_list[self.tabWidget.currentIndex()].enableAutoRange(axis = "x")
        self.graph_rangeMin[self.tabWidget.currentIndex()] = 0
        self.graph_rangeMax[self.tabWidget.currentIndex()] = 1000
        
    def zoom_in_out(self,change):
        ##zooming in by changing x-axis scale
        self.beforeWidget_list[self.tabWidget.currentIndex()].plotItem.getViewBox().scaleBy(x = change, y = 1)
        self.afterWidget_list[self.tabWidget.currentIndex()].plotItem.getViewBox().scaleBy(x = change, y = 1)
    
    def move_right_left(self, move):
        self.afterWidget_list[self.tabWidget.currentIndex()].plotItem.getViewBox().translateBy(x = move)
        self.beforeWidget_list[self.tabWidget.currentIndex()].plotItem.getViewBox().translateBy(x = move)


    def export(self):
        ##create the pdf
        
        self.stop()
        
        ##pdf function
        pdf = FPDF()
        
        ## before equalizing Page
        pdf.add_page()
        ## set pdf title
        pdf.set_font('Arial', 'B', 15)
        pdf.cell(70)
        pdf.cell(60, 10, 'Equalizer Report', 1, 0, 'C')
        pdf.ln(20)
        
        ##take pics of drwan graphs
        exporter1 = pg.exporters.ImageExporter(self.spectroWidget_list[self.tabWidget.currentIndex()].plotItem)
        exporter1.export('spectroAfter.png')
        exporter2 = pg.exporters.ImageExporter(self.beforeWidget_list[self.tabWidget.currentIndex()].plotItem)
        exporter2.export('before.png')
        exporter3 = pg.exporters.ImageExporter(self.afterWidget_list[self.tabWidget.currentIndex()].plotItem)
        exporter3.export('after.png')
        
        ## put before equalizing graph
        pdf.image('before.png', 10, 50, 190, 50)
        os.remove('before.png')
        pdf.image("spectroBefore"+str(self.tabWidget.currentIndex()+1)+".png", 10, 140, 190, 100)
        os.remove("spectroBefore"+str(self.tabWidget.currentIndex()+1)+".png")
        
        ## After equalizing Page
        pdf.add_page()
        ## write the gains of the equalizer
        pdf.set_font('Arial', 'B', 15)
        pdf.cell(90)
        pdf.cell(60, 10, "Gains")
        pdf.ln(10)
        pdf.cell(70)
        pdf.cell(60, 10, str(self.gainArrays[self.tabWidget.currentIndex()]))
        ## put after equalizing graphs
        pdf.image('after.png', 10, 50, 190, 50)
        os.remove('after.png')
        pdf.image("spectroAfter.png", 10, 140, 190, 100)
        os.remove("spectroAfter.png")
        
        ## export Pdf file
        pdf.output("report"+str(self.tabWidget.currentIndex()+1)+".pdf", "F") 
        
        print("Report PDF is ready")
        eqname = name.split('.')[0]
        #save the audio file
        m = np.max(np.abs(self.current_samples[self.tabWidget.currentIndex()]))
        signal = (self.current_samples[self.tabWidget.currentIndex()] / m).astype(np.float32)
        fs = self.sampling_rate_list[self.tabWidget.currentIndex()]
        write(eqname+"_equalized"+".wav", fs, signal)

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = mainwind()
    window.show()
    sys.exit(app.exec_())