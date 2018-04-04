#import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def get_data(filename):
    ''' function to import the data '''
    #Use open() to ensure that the file is closed even if an error occurs.
    with open(filename,'r') as f:
        return pd.read_csv(f,header=None)

def get_fig_title(filename):
    #Can be improved in future for more general use
    return os.path.basename(filename).replace('.csv','')

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    if path_to_dir.endswith('/') == False:
        path_to_dir+='/'
        
    filenames = os.listdir(path_to_dir)
    return [ path_to_dir+filename for filename in filenames if filename.endswith( suffix ) ]

def plot_spectroscopy_signal(full_filename):
    import_data = get_data(full_filename)
    print(full_filename)
    print(import_data.shape)
    #print(import_data[3],import_data[4],import_data[9],import_data[10])
    fig=plt.figure()
    ax=fig.add_subplot(111, label ='1')
    
    #ax3=fig.add_subplot(111, label ='3')
    
    fig_title=get_fig_title(full_filename)
    fig.suptitle(fig_title)
    
    ax.plot(import_data[3],import_data[4], color='b')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage Ch1 [V]", color="b")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y', colors="b")
    
    if import_data.shape[1]>=7:
        ax2=fig.add_subplot(111, label ='2', frame_on=False)
        ax2.plot(import_data[9],import_data[10],color='r')
        #ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        #ax2.set_xlabel('x label 2', color="C1") 
        ax2.set_ylabel('Voltage Ch2 [V]', color='r')       
        #ax2.xaxis.set_label_position('top') 
        ax2.yaxis.set_label_position('right') 
        #ax2.tick_params(axis='x', colors="C1")
        ax2.tick_params(axis='y', colors='r')
        
    plt.savefig(os.path.dirname(full_filename)+'/'+fig_title, format='pdf')



#filepath='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/spektroskopie_cool/'
filepath='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/spektroskopie_RP/'


filelist=find_csv_filenames(filepath)
print(filelist)
for f in filelist:
    plot_spectroscopy_signal(f)



#print(import_data[3])
#For CSV-files from TDS 1001B, the Datapoints are in columns 4,5 (Channel 1) and 10,11 (Channel 2)




