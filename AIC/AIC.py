import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#AIC function calculator
def calc_AIC(waveform):
    np.seterr(divide='ignore', invalid='ignore') #supresses warnings for edge cases (safe)
    AIC = np.zeros(len(waveform))
    k = 0.00000001 #softening coefficient to avoid numerical instability 
    for i in range(len(waveform)):
        AIC[i] = i * np.log10(np.var(waveform[:i]) + k) + (len(waveform) - i - 1) * np.log10(np.var(waveform[i:]) + k)
    
    return AIC

#plotter function
def my_plotter(waveform, AIC):
    Ts = 0.0000002000 #sampling time
    x = np.arange(0, len(waveform)) * Ts 
    #setting figure and plotting
    fig, ax = plt.subplots()

    l1, = ax.plot(x, waveform) 
    ax1 = ax.twinx() 
    l2, = ax1.plot(x, AIC,'C1') 
    #setting up figure axes and data 
    ax.set_xlabel('Time (sec)')  
    ax.set_ylabel('Amplitude (V)')  
    ax1.set_ylabel('AIC') 
    ax.set_title("AIC picker for an AE Signal")  
    ax1.legend([l1, l2],['AE Signal', 'AIC Function']) 

    ax.grid(True)
    fig.tight_layout()
    plt.show() 

#reading the waveform
waveform_1 = np.array(pd.read_csv("line0_1_1_5409066.csv", skiprows = 11)) 
waveform_2 = np.array(pd.read_csv("line0_1_2_11194167.csv", skiprows = 11))

my_plotter(waveform_1, calc_AIC(waveform_1))
my_plotter(waveform_2, calc_AIC(waveform_2))




