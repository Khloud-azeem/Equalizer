import math
import wave
import struct

def make_sine(freq=440, datasize=10000, fname="test.wav", framerate=44100.00):
    amp = 800.0 # amplitude
    sine_list=[]
    for x in range(datasize):
        sine_list.append(math.sin(2*math.pi * (x+1) * freq * ( x/frate)))
    # Open up a wav file
    wav_file=wave.open(fname,"w")
    # wav params
    nchannels = 1
    sampwidth = 2
    framerate = int(frate)
    nframes=datasize
    comptype= "NONE"
    compname= "not compressed"
    wav_file.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    #write on file
    for s in sine_list:
        wav_file.writeframes(struct.pack('h', int(s*amp/2)))
    wav_file.close()

frate = 44100 #that's the framerate
freq=300 #that's the frequency, in hertz
seconds = 3 #seconds of file
data_length = frate*seconds #number of frames
fname = "output.wav" #name of file
make_sine(freq, data_length, fname) 