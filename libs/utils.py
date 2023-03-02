
import numpy as np
import wave

import cv2
from cv2 import VideoCapture
import matplotlib.pyplot as plt


x_lines, y_lines = [], []


def img2fre(img):
    return np.fft.fft2(img)

def visualize_fre(fre):
    fshift = np.fft.fftshift(fre)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum

def genSoundFromImage(file, 
                    output="sound.wav", 
                    duration=5.0, 
                    sampleRate=44100.0, 
                    intensityFactor=1, 
                    min_freq=0, 
                    max_freq=22000, 
                    invert=False, 
                    contrast=True, 
                    highpass=True, 
                    verbose=False):

    wavef = wave.open(output,'w')
    wavef.setnchannels(1) # mono
    wavef.setsampwidth(2) 
    wavef.setframerate(sampleRate)
    
    max_frame = int(duration * sampleRate)
    max_intensity = 32767 # Defined by WAV
    
    stepSize = 400 # Hz, each pixel's portion of the spectrum
    steppingSpectrum = int((max_freq-min_freq)/stepSize)
    
    imgMat = loadPicture(size=(steppingSpectrum, max_frame), file=file, contrast=contrast, highpass=highpass, verbose=verbose)
    if invert:
        imgMat = 1 - imgMat
    imgMat *= intensityFactor # To lower/increase the image overall intensity
    imgMat *= max_intensity # To scale it to max WAV audio intensity
    if verbose:
        print("Input: ", file)
        print("Duration (in seconds): ", duration)
        print("Sample rate: ", sampleRate)
        print("Computing each soundframe sum value..")
    for frame in range(max_frame):
        if frame % 60 == 0: # Only print once in a while
            print("Progress: ==> {:.2%}".format(frame/max_frame), end="\r")
        signalValue, count = 0, 0
        for step in range(steppingSpectrum):
            intensity = imgMat[step, frame]
            if intensity < 0.1*intensityFactor:
                continue
            # nextFreq is less than currentFreq
            currentFreq = (step * stepSize) + min_freq
            nextFreq = ((step+1) * stepSize) + min_freq
            if nextFreq - min_freq > max_freq: # If we're at the end of the spectrum
                nextFreq = max_freq
            for freq in range(currentFreq, nextFreq, 1000): # substep of 1000 Hz is good
                signalValue += intensity*np.cos(freq * 2 * np.pi * float(frame) / float(sampleRate))
                count += 1
        if count == 0: count = 1
        signalValue /= count
        
        data = np.pack('<h', int(signalValue))
        wavef.writeframesraw( data )
        
    wavef.writeframes(''.encode())
    wavef.close()
    print("\nProgress: ==> 100%")
    if verbose:
        print("Output: ", output)


def videoRead(file_path) -> VideoCapture:
    video = cv2.VideoCapture(file_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))    
    return video, (width, height, frames_per_second, num_frames)

def runOnVideo(video, sample_size= (512,256), max_hz=44100.0) -> None:
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """
    readFrames = 0 
    # Frequency Graph
    smaple_size = sample_size[0] * sample_size[1]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1) 
    ax1.set_title("X Frequency")
    ax1.set_ylim(ymin=-max_hz, ymax=max_hz)
    lineX = ax1.plot(np.arange(smaple_size),  np.ones_like(np.arange(smaple_size)), 'b-')[0]
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_ylim(ymin=-max_hz, ymax=max_hz)
    lineY = ax2.plot(np.arange(smaple_size), np.ones_like(np.arange(smaple_size)), 'r-')[0]
    

    while True:
        
        hasFrame, frame = video.read()

        if not hasFrame:
            video.set(cv2.CAP_PROP_POS_MSEC, 0.0)
            hasFrame, frame = video.read()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, sample_size)
        frequency = img2fre(frame)
        vis_fre   = visualize_fre(frequency)
        vis_fre   = cv2.cvtColor(vis_fre.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        music_fre = ((vis_fre/255.) - 0.5) * max_hz
        
    
        # Make 1D Plot and Draw
        x_line = music_fre.flatten()
        y_line = music_fre.flatten('F')
        lineX.set_ydata(x_line)
        lineY.set_ydata(y_line)

        

        fig.canvas.draw()
        plotimg = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,  sep='')
        plotimg = plotimg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plotimg = cv2.cvtColor(plotimg,cv2.COLOR_RGB2BGR)

        x_lines.append(x_line)
        y_lines.append(y_line)


        readFrames += 1
        yield frame, vis_fre, plotimg, music_fre


if __name__ == "__main__":
    out = cv2.VideoWriter('freout.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 4, (512, 256))
    for pa in ["data\\img/start.gif", "data\\img/first.gif"]:
        video, (width, height, frames_per_second,_) = videoRead(pa)
        print(frames_per_second)

        vs =[]
        fs = []
        ms = []
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        

        for v,f, p, m  in runOnVideo(video):
        
            cv2.imshow('AI Said',v)
            cv2.imshow('AI Frequency',f)
            cv2.imshow("AI Frequency Plot", p)

            out.write(v)
            vs.append(v)
            fs.append(f)
            ms.append(m)    

        with wave.open(f"{pa.split('/')[-1].replace('.gif', '')}testFrame.wav", 'w') as wavef:
            wavef.setnchannels(3) # trio
            wavef.setsampwidth(2) 
            wavef.setframerate(512 * 128)
            print(np.shape(vs))
            vs = np.concatenate(vs, axis=0)
            wavef.writeframesraw(vs)
            

        with wave.open(f"{pa.split('/')[-1].replace('.gif', '')}testFrequency.wav", 'w') as wavef:
            wavef.setnchannels(1) # mono
            wavef.setsampwidth(2) 
            wavef.setframerate(512 * 128)
            print(np.shape(fs))
            fs = np.concatenate(fs, axis=0)
            wavef.writeframesraw(fs)
            print(fs.shape)

        with wave.open(f"{pa.split('/')[-1].replace('.gif', '')}testMusic.wav", 'w') as wavef:
            wavef.setnchannels(1) # mono
            wavef.setsampwidth(2) 
            wavef.setframerate(512 * 1024)
            print(np.shape(ms))
            ms = np.concatenate(ms, axis=0)
            wavef.writeframesraw(ms)
            print(ms.shape)
    
    cv2.destroyAllWindows()