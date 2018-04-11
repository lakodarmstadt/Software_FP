import numpy as np
from scipy import misc
import scipy.constants
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def find_bmp_filenames( path_to_dir, suffix=".bmp"):
    if path_to_dir.endswith('/') == False:
        path_to_dir+='/'
        
    filenames = os.listdir(path_to_dir)
    return [ path_to_dir+filename for filename in filenames if filename.endswith( suffix ) ]

def gaussian_1D(height, center, width, offset):
    width=float(width) #width is defined as 1/e^2 of the intensity
    return lambda x: height*np.exp(-2*((center-x)/width)**2)+offset

def gaussian_2D(height, center_x, center_y, width_x, width_y,offset):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-2*(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2))+offset

def data_1D_av(data,x,y):
    data_x=np.sum(data,axis=0)
    data_y=np.sum(data,axis=1)
    return data_x,data_y
   
def moments_1D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 1D distribution by calculating its
    moments """
    total=data.sum()
    X=np.indices(data.shape)
    x = (X*data).sum()/total
    width = np.sqrt(np.abs((np.arange(data.size)-x)**2*data).sum()/data.sum())
    height = data.max()
    offset=data.min()
    return height,x,width,offset
    

def moments_2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
#    #Find mass centre of intensity distribution
#    x = (X*data).sum()/total
#    y = (Y*data).sum()/total
    #Use position of max. values for x and y:
    x,y = np.unravel_index(np.argmax(data, axis=None), data.shape)    
    
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    offset=data.min()
    return height, x, y, width_x, width_y,offset

def fitgaussian_1D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 1D distribution found by a fit"""
    params = moments_1D(data)
    print(params)

    #gaussian_2D(*p) is a function taking two arguments, which are in the case used here two 2D matrices created by np.indices.
    # The error function is then a function depending on the parameters p which gets least if gaussian_2D best fits data.   
    errorfunction = lambda p: np.ravel(gaussian_1D(*p)(*np.indices(data.shape))-data)
                                
    p, pcov, infodict, errmsg, success = optimize.leastsq(errorfunction, params, full_output=1)
    #print('Gaussian_2D:')
    #print(gaussian_2D(*p))
    #print('Product:')
    #print((*np.indices(data.shape)))
    #print(gaussian_2D(*p)(*np.indices(data.shape)))
    #print('Data:')
    #print(data)
    
    return p, pcov, success

def fitgaussian_2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments_2D(data)
    print(params)

    #gaussian_2D(*p) is a function taking two arguments, which are in the case used here two 2D matrices created by np.indices.
    # The error function is then a function depending on the parameters p which gets least if gaussian_2D best fits data.   
    errorfunction = lambda p: np.ravel(gaussian_2D(*p)(*np.indices(data.shape))-data)
                                
    p, pcov, infodict, errmsg, success = optimize.leastsq(errorfunction, params, full_output=1)
    #print('Gaussian_2D:')
    #print(gaussian_2D(*p))
    #print('Product:')
    #print((*np.indices(data.shape)))
    #print(gaussian_2D(*p)(*np.indices(data.shape)))
    #print('Data:')
    #print(data)
    
    return p, pcov, success



def fit_gaussian_2D_to_image(data,filename, pixelsize_x=6.7, pixelsize_y=6.7, sliceing=None):
    #data = np.array(misc.imread(filename))
    if sliceing is not None:
        data = data[sliceing[0][0]:sliceing[0][1], sliceing[1][0]:sliceing[1][1]]

    plt.matshow(data, cmap=plt.cm.gist_earth_r)

    params, pcov, success = fitgaussian_2D(data)
    fit = gaussian_2D(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, width_x, width_y,offset) = params
    print('2D params are:')
    print(params)
    print(np.sqrt(np.diag(pcov)))
    print(success)
    plt.text(0.95, 0.05, """
    width_x : %.1fum
    width_y : %.1fum""" %(width_x*pixelsize_x, width_y*pixelsize_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    plt.text(0.5, 0.9, filename,
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    plt.savefig(filename+'.pdf', format='pdf')
    plt.close()

def fit_gaussian_1D_to_image(data,filename, pixelsize_x=6.7, pixelsize_y=6.7, lin=True):
    #data = np.array(misc.imread(filename))

    gs = gridspec.GridSpec(3,2,height_ratios=[2,2,1])
    
    fig = plt.figure(figsize=(10,10),tight_layout=True)
    
    ax1 = plt.subplot(gs[0:2, :]) #row 0, span all columns
    ax1.matshow(data, cmap=plt.cm.gist_earth_r)

    params, pcov, success = fitgaussian_2D(data)
    fit = gaussian_2D(*params)

    ax1.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    #axtext = ax1.gca()
    (height, x, y, width_x, width_y,offset) = params
    print('2D params are:')    
    print(params)
    print(np.sqrt(np.diag(pcov)))
    print(success)
    ax1.text(0.95, 0.05, """
    width_x : %.1fum
    width_y : %.1fum""" %(width_x*pixelsize_x, width_y*pixelsize_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax1.transAxes)
    ax1.text(0.5, 0.9, filename,
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax1.transAxes)
    if lin==True:
        data_x,data_y=data[int(np.round(x,0)),:],data[:,int(np.round(y,0))]
        filename+='_1D_lin_'+'.pdf'
    else:
        data_x,data_y=data_1D_av(data,x,y)
        filename+='_1D_avg_'+'.pdf'
    
    
    params_x, pcov_x, success_x=fitgaussian_1D(data_x)
    params_y, pcov_y, success_y=fitgaussian_1D(data_y)
    print('1D_x params are:')
    print(params_x)
    print('1D_y params are:')
    print(params_y)
    width_x_1D,width_y_1D=params_x[2],params_y[2]
    
    fit_x=gaussian_1D(*params_x)
    fit_y=gaussian_1D(*params_y)
    
    ax2 = plt.subplot(gs[2, 0]) # row 1, col 0
    ax2.plot(data_x,'ro')    
    ax2.plot(fit_x(*np.indices(data_x.shape)))
    ax2.text(0.95, 0.05, """
    width_x : %.1fum""" %(width_x_1D*pixelsize_x),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax2.transAxes)

    ax3 = plt.subplot(gs[2, 1]) # row 1, col 1
    ax3.plot(data_y,'ro')
    ax3.plot(fit_y(*np.indices(data_y.shape)))
    ax3.text(0.95, 0.05, """
    width_y : %.1fum""" %(width_y_1D*pixelsize_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax3.transAxes)
    
    plt.savefig(filename, format='pdf')
    plt.close()




def data_processing(data_files, bg_file):
    dim_x,dim_y=np.shape(np.array(misc.imread(data_files[0])))
    raw_data= np.empty((dim_x,dim_y,len(data_files)))
    for i in range(len(data_files)):
        raw_data[:,:,i]=np.array(misc.imread(data_files[i]))
    BG_data=np.array(misc.imread(bg_file))
    
    #Convert background data into 3D array:
    BG_data=BG_data[:,:,None]
#    print(raw_data.shape)
    print(BG_data.shape)
    
    # Subtract background image from data. Define data type to unsure that substraction works correctly (otherwise problems with uint8 dtype)
    data =np.subtract(raw_data,BG_data, dtype=float)
    data = np.clip(data,0,255) # Make all negative values 0
    print(data.shape)
    print(raw_data[10,0,:])
    print(BG_data[10,0])
    print(data[10,0,:])
    
    return data

def calc_Temp(sigma_v_squared):
    m=85*1.66*1e-27
    k_B=scipy.constants.k
    T=(m/k_B)*sigma_v_squared
    return T

def plot_MOT_Load(data):
    
    #Sum intensities of all files seperately and plot them vs. image number/time step 
    summed_data=np.sum(data,axis=(0,1))
    plt.plot(np.arange(20),summed_data)
    plt.xlabel('Mot_Loading_time')
    plt.show()
    
def plot_Temp(data,filename,camera_pixel_factor_x=6.7,camera_pixel_factor_y=6.7):
    width=np.empty((2,data.shape[2]))
    for i in range(data.shape[2]):
        p,pcov,success=fitgaussian_2D(data[:,:,i])
        fit_gaussian_2D_to_image(data[:,:,i],filename+str(i))        
        #print(p,success)
        width_x,width_y=p[3],p[4]
        #print(width_x,width_y)
        width[:,i]=[width_x*camera_pixel_factor_x,width_y*camera_pixel_factor_y]
    print(width)
    #Fit the function sigma_t^2=A+(k_b*T/m)*t^2:    
    p_x=np.polyfit(np.square(np.arange(5)*0.25),np.square(width[0,:]),1)
    p_y=np.polyfit(np.square(np.arange(5)*0.25),np.square(width[1,:]),1)
    print(p_x,p_y)
    
    #Calculate the Temperature in mK (Factor 1e-6 because slope is in um^2/ms^2):
    Temp_x=calc_Temp(p_x[0]*1e-3)
    Temp_y=calc_Temp(p_y[0]*1e-3)
    print(Temp_x,Temp_y)
    
    plt.plot(np.square(np.arange(5)*0.25),np.square(width[0,:]))
    plt.plot(np.square(np.arange(5)*0.25),np.square(np.arange(5)*0.25)*p_x[0]+p_x[1])
    plt.plot(np.square(np.arange(5)*0.25),np.square(width[1,:]))
    plt.plot(np.square(np.arange(5)*0.25),np.square(np.arange(5)*0.25)*p_y[0]+p_y[1])
    plt.xlabel('TOF^2[ms^2]')
    plt.ylabel('width^2 [um^2]')
    plt.title('T_x is '+str(Temp_x)+' ;Temp_y is '+str(Temp_y))
    #plt.show()
    plt.savefig(filename+'_eval', format='pdf')
    
    


##
##
##MOT Ladephase:
#
#filepath_MOT='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/MOT_Ladephase'   
##print(find_bmp_filenames(filepath))
#MOT_files=[filename for filename in find_bmp_filenames(filepath_MOT) if 'MOT_Load' in filename]
BG_file=[filename for filename in find_bmp_filenames(filepath_MOT) if 'BG' in filename][0]
#
##Sort the MOT_files correctly
#MOT_files_split=[f.split(' ') for f in MOT_files]
#MOT_files_sorted=[' '.join(f) for f in sorted(MOT_files_split,key=lambda x: x[1])]
##print(MOT_files_sorted)
#
#plot_MOT_Load(data_processing(MOT_files_sorted, BG_file))
#
##
##
##
##MOT_Temp:
#
#filepath_MOT_Temp='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/MOT_Temperatur'
#MOT_Temp_files=[filename for filename in find_bmp_filenames(filepath_MOT_Temp)]
##Sort the MOT_files correctly
#MOT_Temp_files_split=[f.split(' ') for f in MOT_Temp_files]
#MOT_Temp_files_sorted=[' '.join(f) for f in sorted(MOT_Temp_files_split,key=lambda x: x[1])]
#
##Use only the first 4 pictures as programm did error:
#MOT_Temp_files_sorted=MOT_Temp_files_sorted[:5]
#print(MOT_Temp_files_sorted)
#MOT_Temp_data=data_processing(MOT_Temp_files_sorted,BG_file)
#print(MOT_Temp_data.shape[2])
#plot_Temp(MOT_Temp_data,filepath_MOT_Temp+'/MOT_Temp')
#
#

#
#
#
#Melasse_Temp:

filepath_Melasse_Temp='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/Melasse_Temperatur/Melasse_5ms'
Melasse_Temp_files=[filename for filename in find_bmp_filenames(filepath_Melasse_Temp)]
#Sort the MOT_files correctly
Melasse_Temp_files_split=[f.split(' ') for f in Melasse_Temp_files]
Melasse_Temp_files_sorted=[' '.join(f) for f in sorted(Melasse_Temp_files_split,key=lambda x: x[1])]

#Use only the first 4 pictures as programm did error:
Melasse_Temp_files_sorted=Melasse_Temp_files_sorted[:5]
print(Melasse_Temp_files_sorted)
Melasse_Temp_data=data_processing(Melasse_Temp_files_sorted,BG_file)
print(Melasse_Temp_data.shape[2])
plot_Temp(Melasse_Temp_data,filepath_Melasse_Temp+'/Melasse_Temp')
