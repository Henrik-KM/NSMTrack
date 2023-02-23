#%%
import matplotlib.ticker as plticker
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from scipy.io import loadmat
from utils.YOLO import manYOLO,manYOLOSplit, ConvertYOLOLabelsToCoord, ConvertCoordToYOLOLabels
from utils.funcs import remove_stuck_particle_2, load_all_models, predict_function, manTrack, CalculateiOCandD
import warnings
#Code which suppresses RunTimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
%matplotlib auto


sys.path.append("deeptrack/")
from deeptrack.models import resnetcnn
from keras import models

def _compile(model: models.Model, 
            *,
            loss="mae", 
            optimizer="adam", 
            metrics=[],
            **kwargs):
    ''' Compiles a model.

    Parameters
    ----------
    model : keras.models.Model
        The keras model to interface.
    loss : str or keras loss
        The loss function of the model.
    optimizer : str or keras optimizer
        The optimizer of the model.
    metrics : list, optional
    '''

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model



def reload_resnet():
    resnet=resnetcnn(input_shape=(None, None, 1),
            conv_layers_dimensions=(16, 32, 64, 128, 256), # sets downsampling size
            upsample_layers_dimensions=(64, 128),
            base_conv_layers_dimensions=(128, 128),
            output_conv_layers_dimensions=(16, 16),
            dropout=(),#0.01,
            pooldim=2,
            steps_per_pooling=1,
            number_of_outputs=1,
            output_activation=None,
            loss="mae",
            layer_function=None,
            BatchNormalization=False,
            conv_step=None)
    return resnet

#Load models
unet_path = "Network-weights/U-Net- I0.01-25_loss_0.009943331.h5" #Normal segmentation model
unet_path_EV="Network-weights/U-Net-I0.02-100_loss_0.005466693_EV.h5" #Segmentation model used for high-iOC particles in large channels
#unet_path="U-net-D0.1-2I0.0-2loss=0.0046777986.h5"
#unet_path="Network-weights/U-net-D0.55-1.3I0.01-10loss0.0028770813.h5"
unet_path=unet_path_EV
resnet_diffusion_path='Network-weights/resnet-diffusion-09062021-0734012048x128_combinedLoss.h5' #Normal diffusion model
resnet_diffusion_path_EV='Network-weights/resnet-diffusion-21072021-054520512x128_EV_combinedLoss.h5' #EV diff model

resnet_intensity_path='Network-weights/resnet-D0.1-1.15 I0.01-25 128x128_loss_5.546048.h5' # Isn't this an old model?? Normal int model
resnet_intensity_path='Network-weights/resnet-D0.01-1 I0.01-30 512x128_loss_5.4842668.h5'
resnet_intensity_path_EV = "Network-weights/resnet-D0.05-1.15 I0.5-1000 512x512_loss_104.02502.h5" #EV int model

resnet_intensity=reload_resnet()
resnet_diffusion=reload_resnet()

unet = tf.keras.models.load_model(unet_path,compile=False)
resnet_intensity.load_weights(resnet_intensity_path)
resnet_diffusion.load_weights(resnet_diffusion_path)

resnet_ensemble = load_all_models()


#%% Enter user variables

#Path to 
mainDataPath='C:/Users/ccx55/OneDrive/Documents/GitHub/Phd/Biosensing---nanochannel-project/Data/Preprocessed degP/2022-12-22_chip220621w02_oo-burm_NEW-HENRIK/Protein-temp-x120/'
#Experimental or simulated data?
experimental_data=True
#EV data?
exosomes=0
#Plot data?
plot=1
#Save images?
save = 0
savePath="C:/Users/ccx55/OneDrive/Documents/GitHub/Phd/Biosensing---nanochannel-project/Figures/degP/2022-12-22_chip220621w02_oo-burm_NEW-HENRIK/Protein-temp-x120/"
if save:
    os.makedirs(savePath, exist_ok=True)

includeSA=1

if exosomes:
    treshold = 0.4
    trajTreshold=8
else:
    treshold=0.05
    trajTreshold=256
#%%    
plt.close('all')
try:
    del intensityArray
    del diffusionArray
    del countarray
except:
    pass

if exosomes:
    resnet_intensity.load_weights(resnet_intensity_path_EV)
    resnet_diffusion.load_weights(resnet_diffusion_path_EV)
    unet = tf.keras.models.load_model(unet_path_EV,compile=False)
else:
    unet = tf.keras.models.load_model(unet_path,compile=False)
    resnet_intensity.load_weights(resnet_intensity_path)
    resnet_diffusion.load_weights(resnet_diffusion_path)
    
# Get a list of all directories in main data path
measurements = os.listdir(mainDataPath)

# Set a maximum number of files to process
maxFiles = 1000

# Use GPU for processing
with tf.device('/GPU:0'):

    # Iterate through each directory in measurements
    for meas in measurements:
        print("Running measurement: " + meas)
        
        # Get the path to the data for this measurement
        dataPath = mainDataPath + "/" + meas
        
        # Get a list of all files in the intensity subdirectory
        files = os.listdir(dataPath + "/intensity/")
        

        SA = os.listdir(dataPath)
        SA = [f for f in SA if ".mat" in f]
        # If there is a .mat file in the measurement directory for standard analysis SA, load it
        if SA != []:
            SA = loadmat(dataPath + "/" + SA[0])
        
        # Limit the number of files to process
        files = files[:maxFiles]
        
        # Iterate through each file to be analyzed
        for fileName in files:
            print("Analyzing File: " + fileName)
            
            # Reset YOLO labels and coordinates if they exist
            try:
                del YOLOLabels
                del YOLOCoords
            except:
                pass
            
            # Load the intensity and diffusion files for prediction
            file = np.load(dataPath + "/intensity/" + fileName)
            diff_file = np.load(dataPath + "/diffusion/" + fileName)
            
            # Add a dimension to the intensity and diffusion files for the network
            file = np.expand_dims(file, 0)
            file = np.expand_dims(file, -1)
            diff_file = np.expand_dims(diff_file, 0)
            diff_file = np.expand_dims(diff_file, -1)
            
            # Get the length and times of the file
            length = file.shape[2] 
            times = file.shape[1]
            img_size = 256
            
            # Determine the time crop (important for e.g. images with camera stuttering flaws)
            timesLimit = times % img_size
            
            # Determine the length cut off based on the length of the file
            if length == 150:
                lengthCutOff = int((150-128)/2)
            elif length == 640:
                lengthCutOff = int((640-512)/2)
            
            # Manage the data based sizes to be consistent
            if not experimental_data:
                orig_img = file[:, 904:904+8192, lengthCutOff:-lengthCutOff, :]
                orig_img_diff = diff_file[:, 904:904+8192, lengthCutOff:-lengthCutOff, :]
            elif timesLimit > 0:
                if file.shape[1] >= 8192+904:
                    orig_img = file[:, 904:904+8192, lengthCutOff:-lengthCutOff, :]
                    orig_img_diff = diff_file[:, 904:904+8192, lengthCutOff:-lengthCutOff, :]
                else:
                    orig_img = file[:, 0:-int(timesLimit), lengthCutOff:-lengthCutOff, :]
                    orig_img_diff = diff_file[:, 0:-int(timesLimit), lengthCutOff:-lengthCutOff, :]
            else:
                orig_img = file[:, :, lengthCutOff:-lengthCutOff, :]
                orig_img_diff = diff_file[:, :, lengthCutOff:-lengthCutOff, :]
            
            # Use the unet model to segment the images
            pred_diff = unet.predict(orig_img_diff)
    
            length = orig_img.shape[2]
            times = orig_img.shape[1]
    
            if plot:
                fig,axs=plt.subplots(1,2,figsize=(16,16))
                ax2 = axs[1]
                ax=axs[0]
                im=ax.imshow(pred_diff[0,:,:,0],aspect='auto')
                plt.colorbar(im,ax=ax)
                ax.set_ylabel('t')
                ax.set_title("Segmented Image")
                ax.set_xlabel('x')
                
                loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
                locY = plticker.MultipleLocator(base=100.0)
                axs[1].xaxis.set_major_locator(loc)
                axs[1].yaxis.set_major_locator(locY)
               
                
                file = np.squeeze(file,0)
                file = np.squeeze(file,-1)
    
    
                im=ax2.imshow(orig_img[0,:,:,0],aspect='auto')
                plt.colorbar(im,ax=ax2)
                ax2.set_ylabel('t')
                ax2.set_title(fileName)
                ax2.set_xlabel('x')        
            #Get YOLO labels from the segmented image
            YOLOLabels = manYOLO(pred_diff,trajTreshold=trajTreshold,treshold=treshold)
            #Split the YOLO labels into individual trajectories
            YOLOLabels = manYOLOSplit(pred_diff,treshold=treshold,trajTreshold=int(trajTreshold/8),splitTreshold=trajTreshold)
        
            if not None in YOLOLabels:
                boxes = ConvertYOLOLabelsToCoord(YOLOLabels[0,:,:],pred_diff.shape[1],pred_diff.shape[2])
    
                for i in range(0,len(boxes)):
                    box=boxes[i,:]
                    _,x1,y1,x2,y2 = box
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    box_l = x2-x1
                    box_t = y2-y1
                    img = pred_diff[:,y1:y2,x1:x2,:]
                    labels=manYOLO(np.transpose(img,(0,2,1,3)))
    
                    labels[:,:,1],labels[:,:,2],labels[:,:,3],labels[:,:,4] = np.copy(labels[:,:,2]), np.copy(labels[:,:,1]), np.copy(labels[:,:,4]), np.copy(labels[:,:,3])
    
                    if not None in labels:
                        coords = ConvertYOLOLabelsToCoord(labels[0,:,:],box_t,box_l)
                        coords[:,1] = coords[:,1]+x1
                        coords[:,2] = coords[:,2]+y1
                        coords[:,3] = coords[:,3]+x1
                        coords[:,4] = coords[:,4]+y1
                        labels = ConvertCoordToYOLOLabels(coords,pred_diff.shape[1],pred_diff.shape[2]).reshape(1,-1,5)
                        try:
                            YOLOCoords = np.append(YOLOCoords,coords,0)
                            YOLOLabels = np.append(YOLOLabels,labels,1)
                        except:
                            YOLOLabels = np.copy(labels)
                            YOLOCoords = np.copy(coords)
    
       
            YOLOLabels=YOLOLabels[0,:,:]
            
            
    
            if not None in YOLOLabels:
    
                detections = ConvertYOLOLabelsToCoord(YOLOLabels,pred_diff.shape[1],pred_diff.shape[2])
                detections = detections[:,1:]
                
                try:
                    detections
                except:
                    detections = np.zeros((0,4))
            
                
                for x1, y1, x2, y2 in detections:
                    x1 = np.max([x1,0])
                    y1 = np.max([y1,0])
                    x2 = np.min([x2,orig_img.shape[2]])
                    y2 = np.min([y2,orig_img.shape[1]])
                    box_w = x2 - x1
                    box_h = int(y2 - y1)    
    
                    yolo_img = orig_img[:,int(y1):int(y2),int(x1):int(x2),:]
                    yolo_img_diff = pred_diff[:,int(y1):int(y2),int(x1):int(x2),:]
                    
                    if yolo_img.shape[1] >= trajTreshold and yolo_img.shape[2] >= int(trajTreshold/8):
                        
    
                        M = 10
                        nbr_its = 5
    
                        if False:
                            yolo_img_diff = remove_stuck_particle_2(yolo_img_diff,M,nbr_its)
                            
    
                        intensity = resnet_intensity(yolo_img,training=False)[0][0][0]
                        diffusion = resnet_diffusion(yolo_img_diff,training=False)[0][0][0]**2*57
                        trajectories,currentTrajectories,_=manTrack(yolo_img_diff[0,:,:,0],trajTreshold=trajTreshold,threshold=treshold)
  
                        image=yolo_img[0,:,:,0]
                        intensitySA, diffusionSA,trajectoryLengths = CalculateiOCandD(trajectories,currentTrajectories,image,plot=False)
                        intensitySA*=10 #Converting to ML units
                    else:
                        intensity = 0
                        diffusion = 0
                        box_h=1
    
                    if plot:
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, zorder=2,edgecolor="black", facecolor="none")
    
                       # if int(cls_pred) == 0:
                        ax2.text(x1,y1,"I1 = "+str(np.round(intensity,2))+", D1 = "+str(np.round(diffusion,1))  + ", I2 = "+str(np.round(intensitySA,2))+", D2 = "+str(np.round(diffusionSA,1)),color = "black")
                        ax2.add_patch(bbox)
                        if save:
                            plt.savefig(savePath+fileName[:-8])
    
                    try:
                        intensityArray = np.append(intensityArray,intensity)  
                        diffusionArray = np.append(diffusionArray,diffusion)  
                        countArray = np.append(countArray,box_h)
                        
                        intensityArraySA = np.append(intensityArraySA,intensitySA)  
                        diffusionArraySA = np.append(diffusionArraySA,diffusionSA)  
                        countArraySA = np.append(countArraySA,trajectoryLengths)
    
                    except:
                        intensityArray = np.array([intensity])
                        diffusionArray = np.array([diffusion])
                        countArray = np.array([box_h])
                        
                        intensityArraySA = np.array([intensitySA])
                        diffusionArraySA = np.array([diffusionSA])
                        countArraySA = np.array([trajectoryLengths])

#%% Convert to histogram and check for complementary SA results

intensityArrayFull = np.repeat(intensityArray, countArray)
diffusionArrayFull = np.repeat(diffusionArray, countArray)
fullIntensitySA = np.repeat(intensityArraySA, countArraySA)
fullDiffusivitySA = np.repeat(diffusionArraySA, countArraySA)

IFull=[]
DFull=[]
if SA!=[]:
    N=SA["collection"][0][0][0][0]
    D=SA["collection"][0][0][2][0]
    I=SA["collection"][0][0][3][0]
    for i in range(0,len(N)):
        IFull=np.append(IFull,[I[i]]*N[i])
        DFull=np.append(DFull,[D[i]]*N[i])
else:
    print("No SA found")
    D=[]
    I=[]
#%%Plot histogram
plt.close('all')
keepInds=np.where(intensityArrayFull>0)[0]
keepInds=np.append(keepInds,np.where(diffusionArrayFull>0)[0])
intensityArrayFull=intensityArrayFull[keepInds]
diffusionArrayFull=diffusionArrayFull[keepInds]

keepInds=np.where(intensityArrayFull<np.median(intensityArrayFull)+3*np.std(intensityArrayFull))[0]
intensityArrayFull=intensityArrayFull[keepInds]
diffusionArrayFull=diffusionArrayFull[keepInds]

keepInds=np.where(fullIntensitySA<np.median(fullIntensitySA)+3*np.std(fullIntensitySA))[0]
fullIntensitySA=fullIntensitySA[keepInds]
fullDiffusivitySA=fullDiffusivitySA[keepInds]

import statistics
from scipy.optimize import curve_fit
from pylab import hist, exp, sqrt, diag, plot, legend
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)
nbr_sigma = 3

def plotHist(data,SAData,gaussianFit=False):
    
    expected_peak = statistics.median(data)
    expected_width = expected_peak/10
    
    
    nbr_bins = 200 # nbr of bins can be automatized
    expected_height = len(data) # not a very sensitive parameter.
    
    plt.figure(figsize=(12,6))
    y,x,_=hist(data,nbr_bins,alpha=.8,label='ML',color="blue",density=False);
    
    x=(x[1:]+x[:-1])/2 # make len(x)==len(y)
    
    expected=(expected_peak,expected_width,expected_height)
    print('expected peak = {:.2f}'.format(expected_peak))
    print('expected width = {:.2f}'.format(expected_width))
    if gaussianFit:
        try:
            params,cov=curve_fit(gauss,x,y,expected)
            sigma=sqrt(diag(cov))
            plot(x,gauss(x,*params),color='red',lw=2,label='model')
            peak = params[0]
            sigma = abs(params[1]) # sometimes the width is outputted as a negative number...
            
            plt.axvline(peak-nbr_sigma*sigma,label='peak +- {}$\sigma$'.format(nbr_sigma),color='black')
            plt.axvline(peak+nbr_sigma*sigma,color='black')       
        
            print('peak = {:.2f}'.format(peak))
            print('width = {:.2f}'.format(sigma))
        except:
            pass
    
    if SAData!=[]:
        y,x,_=hist(SAData,nbr_bins,alpha=.5,label='SA',color="black",density=False);
    plt.title(mainDataPath)
    legend(fontsize=14)
    
# Expected peak and width parameters can be changed (and automatized more effectively), but the current settings seem to work well.
plotHist(intensityArrayFull,np.array(IFull)*10000)
if save:
    plt.savefig(savePath+"IHist")
    np.savetxt(savePath+"iOC",intensityArrayFull)
plotHist(diffusionArrayFull,DFull)
if save:
    plt.savefig(savePath+"DHist")
    np.savetxt(savePath+"D",diffusionArrayFull)
    
    
plotHist(intensityArrayFull,fullIntensitySA)
if save:
    plt.savefig(savePath+"IHist")
    np.savetxt(savePath+"iOC",intensityArrayFull)
plotHist(diffusionArrayFull,fullDiffusivitySA)
if save:
    plt.savefig(savePath+"DHist")
    np.savetxt(savePath+"D",diffusionArrayFull)
    
    
plotHist(fullIntensitySA,np.array(IFull)*10000)
if save:
    plt.savefig(savePath+"IHist")
    np.savetxt(savePath+"iOC",intensityArrayFull)
    
plotHist(fullDiffusivitySA,DFull)
if save:
    plt.savefig(savePath+"DHist")
    np.savetxt(savePath+"D",diffusionArrayFull)

plt.figure()
uniqueIntensities = np.unique(intensityArrayFull,return_index=True)
uniqueDiffusivities = diffusionArrayFull[uniqueIntensities[1]]
uniqueIntensities= intensityArrayFull[uniqueIntensities[1]]
plt.scatter(uniqueIntensities,uniqueDiffusivities)
#plt.ylim(10,20)
plt.gca().invert_yaxis()

# np.save(meas+"-intensity.npy",intensityArrayFull)
# np.save(meas+"-diffusion.npy",diffusionArrayFull)

