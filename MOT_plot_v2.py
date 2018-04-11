import numpy as np
from scipy import misc
import scipy.constants
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

um=1e-6

def find_bmp_filenames( path_to_dir, suffix=".bmp"):
    if path_to_dir.endswith('/') == False:
        path_to_dir+='/'

    filenames = os.listdir(path_to_dir)
    return [ path_to_dir+filename for filename in filenames if filename.endswith( suffix ) ]

def gaussian_1D(height, center, width, offset):
    width=float(width) #width is defined as 1/e^2 of the intensity
    return lambda x: height*np.exp(-((center-x)/(2*width))**2)+offset

def gaussian_2D(height, center_x, center_y, width_x, width_y,offset):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/(2*width_x))**2+((center_y-y)/(2*width_y))**2))+offset

def data_1D_av(data,i,j):
    data_i=np.sum(data,axis=0)
    data_j=np.sum(data,axis=1)
    return data_i,data_j

def moments_1D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 1D distribution by calculating its
    moments """
    total=data.sum()
    I=np.indices(data.shape)
    i = (I*data).sum()/total
    width = np.sqrt(np.abs((np.arange(data.size)-i)**2*data).sum()/data.sum())
    height = data.max()
    offset=data.min()
    return height,i,width,offset


def moments_2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    I, J = np.indices(data.shape)
    #Find mass centre of intensity distribution
    i = (I*data).sum()/total
    j = (J*data).sum()/total
    col = data[:, int(j)]
    width_i = np.sqrt(np.abs((np.arange(col.size)-i)**2*col).sum()/col.sum())
    row = data[int(i), :]
    width_j = np.sqrt(np.abs((np.arange(row.size)-j)**2*row).sum()/row.sum())
    height = data.max()
    offset=data.min()
    return height, i, j, width_i, width_j,offset

def moments_2D_max(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments; the center values i and j are found by using the coordinates from the maximum value"""
#    total = data.sum()
#    I, J = np.indices(data.shape)
    #Find maximum of intensity distribution
    i,j = np.unravel_index(np.argmax(data, axis=None), data.shape)
    #TODO
    # Use the n biggest values to be more robust against dead pixels:
    col = data[:, int(j)]
    width_i = np.sqrt(np.abs((np.arange(col.size)-i)**2*col).sum()/col.sum())
    row = data[int(i), :]
    width_j = np.sqrt(np.abs((np.arange(row.size)-j)**2*row).sum()/row.sum())
    height = data.max()
    offset=data.min()
    return height, i, j, width_i, width_j,offset

def fitgaussian_1D(data):
    """Returns (height, i, j, width_i, width_j)
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
    """Returns (height, i, j, width_i, width_j)
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



def fit_gaussian_2D_to_image(data, filename, pixelsize_x=6.7, pixelsize_y=6.7, sliceing=None):
    #data = np.array(misc.imread(filename))
    if sliceing is not None:
        data = data[sliceing[0][0]:sliceing[0][1], sliceing[1][0]:sliceing[1][1]]

    #fig = plt.figure(figsize=(5,4),tight_layout=True)
    plt.figure(figsize=(5,4),tight_layout=True)
    plt.matshow(data, cmap=plt.cm.gist_earth_r)
#    plt.matshow(data, cmap='gray')


    params, pcov, success = fitgaussian_2D(data)
    fit = gaussian_2D(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    #ax.xaxis.set_tick_params(labeltop='off', labelbottom='on')
    (height, i, j, width_i, width_j,offset) = params
    print('2D params are:')
    print(params)
    print(np.sqrt(np.diag(pcov)))
    print(success)
    plt.text(0.95, 0.05, """
    width_x : %.1fum
    width_y : %.1fum""" %(width_j*pixelsize_x, width_i*pixelsize_y),
            fontsize=12, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    plt.text(0.5, 0.9, os.path.basename(filename),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    ax.set_xlabel('x-axis camera')
    ax.set_ylabel('y-axis camera')
    #fig.tight_layout()
    plt.savefig(filename + '.pdf', bbox_inches='tight', format='pdf')
    plt.close()

def fit_gaussian_1D_to_image(data, filename, pixelsize_x=6.7, pixelsize_y=6.7, lin=True):


    gs = gridspec.GridSpec(3,2,height_ratios = [2,2,1])

    fig = plt.figure(figsize=(10,10),tight_layout=True)

    ax1 = plt.subplot(gs[0:2, :]) #row 0, span all columns
    ax1.matshow(data, cmap=plt.cm.gist_earth_r)

    params, pcov, success = fitgaussian_2D(data)
    fit = gaussian_2D(*params)

    ax1.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax1.xaxis.set_tick_params(labeltop='off', labelbottom='on')
    ax1.set_xlabel('y-axis camera')
    ax1.set_ylabel('x-axis camera')
    #axtext = ax1.gca()
    (height, i, j, width_i, width_j,offset) = params
    print('2D params are:')
    print(params)
    print(np.sqrt(np.diag(pcov)))
    print(success)
    ax1.text(0.95, 0.05, """
    waist_x : %.1fum
    waist_y : %.1fum""" %(width_j*pixelsize_x, width_i*pixelsize_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax1.transAxes)
    ax1.text(0.5, 0.9, os.path.basename(filename),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax1.transAxes)
    if lin==True:
        data_i,data_j=data[:,int(np.round(j,0))],data[int(np.round(i,0)),:]
        filename+='_1D_lin_'+'.pdf'
    else:
        data_i,data_j=data_1D_av(data,i,j)
        filename+='_1D_avg_'+'.pdf'


    params_i, pcov_i, success_i=fitgaussian_1D(data_i)
    params_j, pcov_j, success_j=fitgaussian_1D(data_j)
    print('1D_x params are:')
    print(params_j)
    print('1D_y params are:')
    print(params_i)
    width_i_1D,width_j_1D=params_i[2],params_j[2]

    # Be aware that i is the row-index specifying the vertical (so the y-) axis of the image. Correspondingly j is the index for the x-axis.
    fit_i=gaussian_1D(*params_i)
    fit_j=gaussian_1D(*params_j)

    ax2 = plt.subplot(gs[2, 0]) # row 1, col 0
    ax2.plot(data_i,'ro')
    ax2.plot(fit_i(*np.indices(data_i.shape)))
    ax2.text(0.95, 0.05, """
    waist_y : %.1fum""" %(width_i_1D*pixelsize_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax2.transAxes)

    ax3 = plt.subplot(gs[2, 1]) # row 1, col 1
    ax3.plot(data_j,'ro')
    ax3.plot(fit_j(*np.indices(data_j.shape)))
    ax3.text(0.95, 0.05, """
    waist_x : %.1fum""" %(width_j_1D*pixelsize_x),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax3.transAxes)

    plt.savefig(filename, format='pdf')
    plt.close()




def data_processing(data_files, bg_file=None):
    # Create an array with the raw input data:
    dim_i,dim_j=np.shape(np.array(misc.imread(data_files[0])))
    raw_data= np.empty((dim_i,dim_j,len(data_files)))
    for i in range(len(data_files)):
        raw_data[:,:,i]=np.array(misc.imread(data_files[i]))

    # Handle background data:
    if bg_file==None:
        BG_data=np.zeros((dim_i,dim_j,1))
        print('No Background file used')
    else:
        BG_data=np.array(misc.imread(bg_file))

        #Convert background data into 3D array:
        BG_data=BG_data[:,:,None]
        #print(raw_data.shape)
    print(BG_data.shape)

    # Subtract background image from data. Define data type to unsure that substraction works correctly (otherwise problems with uint8 dtype)
    data = np.subtract(raw_data,BG_data, dtype=float)
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

def calc_Temp2(sigma_v):
    m=85*1.66*1e-27
    k_B=scipy.constants.k
    T=(m/k_B)*sigma_v**2
    return T

def plot_MOT_Load(data):

    #Sum intensities of all files seperately and plot them vs. image number/time step
    summed_data=np.sum(data,axis=(0,1))
    plt.plot(np.arange(len(summed_data)),summed_data)
    plt.xlabel('Mot_Loading_time')
    plt.show()

def plot_Temp(data,filename,camera_pixel_factor_x=6.7*um,camera_pixel_factor_y=6.7*um):
    width=np.empty((2,data.shape[2]))
    for i in range(data.shape[2]):
        p,pcov,success=fitgaussian_2D(data[:,:,i])
        fit_gaussian_2D_to_image(data[:,:,i],filename+str(i))
        #print(p,success)

        width_i,width_j=p[3],p[4]
        #print(width_i,width_j)
        #Transition from i,j to x,y:
        width[:,i]=[width_j*camera_pixel_factor_x,width_i*camera_pixel_factor_y]
    print(width)

    width_squared=width**2
    print(width_squared)

    TOF_step=0.25*1e-3
    Time_steps=np.arange(len(width[0,:]))*TOF_step
    print(Time_steps)
    Time_squared=Time_steps**2

    #Fit the function sigma_t^2=A+(k_b*T/m)*t^2:
    p_x=np.polyfit(Time_squared,width_squared[0,:],1)
    p_y=np.polyfit(Time_squared,width_squared[1,:],1)
    print(p_x,p_y)

    #Calculate the Temperature in K:
    Temp_x=calc_Temp(p_x[0])
    Temp_y=calc_Temp(p_y[0])
    print(Temp_x,Temp_y)

    plt.plot(Time_squared*1e6,(width_squared[0,:])*1e12)
    plt.plot(Time_squared*1e6,(Time_squared*p_x[0]+p_x[1])*1e12)
    plt.plot(Time_squared*1e6,(width_squared[1,:])*1e12)
    plt.plot(Time_squared*1e6,(Time_squared*p_y[0]+p_y[1])*1e12)
    plt.xlabel('TOF^2[ms^2]')
    plt.ylabel('width^2 [um^2]')
    plt.title('T_x [mK] is '+str(Temp_x*1e3)+' ;T_y [mK] is '+str(Temp_y*1e3))
    #plt.show()
    plt.savefig(filename+'_eval.pdf')
    plt.close()

#
def file_sorter(files, splitter):
    files_split=[f.split(splitter) for f in files]
    files_sorted=[splitter.join(f) for f in sorted(files_split,key=lambda x: x[1])]
    return files_sorted


#
#
##MOT Ladephase:
#
#filepath_MOT='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/MOT_Ladephase'
##print(find_bmp_filenames(filepath))
#MOT_files=[filename for filename in find_bmp_filenames(filepath_MOT) if 'MOT_Load' in filename]
#BG_file=[filename for filename in find_bmp_filenames(filepath_MOT) if 'BG' in filename][0]
#
##Sort the MOT_files correctly
#MOT_files_split=[f.split(' ') for f in MOT_files]
#MOT_files_sorted=[' '.join(f) for f in sorted(MOT_files_split,key=lambda x: x[1])]
#print(MOT_files_sorted)

#plot_MOT_Load(data_processing(MOT_files_sorted, BG_file))




# #MOT_Temp1:

# filepath_MOT_Temp='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/2018_04_05/MOT_Temp'
# BG_file=[filename for filename in find_bmp_filenames(filepath_MOT_Temp) if 'BG' in filename][0]
# Splitter_MOT='MOT_Temp0'
# MOT_Temp_files=[filename for filename in find_bmp_filenames(filepath_MOT_Temp) if Splitter_MOT in filename]
# print(MOT_Temp_files)
# #Sort the MOT_files correctly

# MOT_Temp_files_sorted=file_sorter(MOT_Temp_files,Splitter_MOT)

# #Use only the first 4 pictures as programm did error:
# MOT_Temp_files_sorted=MOT_Temp_files_sorted[:]
# print(MOT_Temp_files_sorted)
# MOT_Temp_data=data_processing(MOT_Temp_files_sorted,BG_file)
# print(MOT_Temp_data.shape[2])
# plot_Temp(MOT_Temp_data,filepath_MOT_Temp+'/MOT_Temp')





# #MOT_Temp2:

# filepath_MOT_Temp='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/MOT_Temperatur'
filepath_BG='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/MOT_Ladephase'
BG_file_2=[filename for filename in find_bmp_filenames(filepath_BG) if 'BG' in filename][0]
# Splitter_MOT='TOF0'
# MOT_Temp_files=[filename for filename in find_bmp_filenames(filepath_MOT_Temp) if Splitter_MOT in filename]
# print(MOT_Temp_files)
# #Sort the MOT_files correctly

# MOT_Temp_files_sorted=file_sorter(MOT_Temp_files,Splitter_MOT)

# #Use only the first 4 pictures as programm did error:
# MOT_Temp_files_sorted=MOT_Temp_files_sorted[:]
# print(MOT_Temp_files_sorted)
# MOT_Temp_data=data_processing(MOT_Temp_files_sorted,BG_file_2)
# print(MOT_Temp_data.shape[2])
# plot_Temp(MOT_Temp_data,filepath_MOT_Temp+'/MOT_Temp')





#Melasse_Temp2:

filepath_Melasse_Temp='/home/lars/Dokumente/Lars_Kohfahl/Studium/PhD/Lehre/FP/Messungen_LK/Melasse_Temperatur/Melasse_10ms'
Splitter_Melasse='Melassezeit0'
Melasse_Temp_files=[filename for filename in find_bmp_filenames(filepath_Melasse_Temp) if Splitter_Melasse in filename]
#Sort the MOT_files correctly

Melasse_Temp_files_sorted=file_sorter(Melasse_Temp_files,Splitter_Melasse)

#Use only the first 4 pictures as programm did error:
Melasse_Temp_files_sorted=Melasse_Temp_files_sorted[:5]
print(Melasse_Temp_files_sorted)
Melasse_Temp_data=data_processing(Melasse_Temp_files_sorted,BG_file_2)
print(Melasse_Temp_data.shape[2])
plot_Temp(Melasse_Temp_data,filepath_Melasse_Temp+'/Melasse_Temp')
