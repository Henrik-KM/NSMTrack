import matplotlib.ticker as plticker
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from utils.YOLO import manYOLO, ConvertYOLOLabelsToCoord, ConvertCoordToYOLOLabels
from utils.funcs import remove_stuck_particle_2,load_all_models,reload_resnet, predict_function

#Load models
unet_path = "Network-weights/U-Net- I0.01-25_loss_0.009943331.h5" #Normal segmentation model
resnet_intensity_path="Network-weights/resnet-D0.01-1 I0.01-30 512x128_loss_5.4842668.h5" #Normal int model
resnet_diffusion_path='Network-weights/resnet-diffusion-09062021-0734012048x128_combinedLoss.h5' #Normal diffusion model

resnet_intensity_base=reload_resnet()
resnet_diffusion_base=reload_resnet()

resnet_intensity_base.load_weights(resnet_intensity_path)
resnet_diffusion_base.load_weights(resnet_diffusion_path)
unet = tf.keras.models.load_model(unet_path,compile=False)

resnet_ensemble = load_all_models()


#%% Enter user variables

#Path to data to be analyzed
mainDataPath='Data/Preprocessed Sample Data/'
#Plot data?
plotImages=False
#Save images?
saveImages = False
#%% Run analysis
plt.close('all')

try:
    del intensityArray
    del diffusionArray
    del intensityEnsembleArray
    del diffusionEnsembleArray
    del meanIntArray
    del meanDiffArray
    del countarray
except:
    pass

measurements = os.listdir(mainDataPath)
for meas in measurements:
    print("Running measurement: "+meas)
    dataPath = mainDataPath+"/"+meas
    files = os.listdir(dataPath + "/intensity/")

    for fileName in files:
        print("Analyzing file: "+fileName)
        try:
            del YOLOLabels
            del YOLOCoords
        except:
            pass

        file = np.load(dataPath+ "/intensity/"+fileName)
        diff_file = np.load(dataPath+ "/diffusion/"+fileName)
        
        file = np.expand_dims(file,0)
        file = np.expand_dims(file,-1)
        diff_file = np.expand_dims(diff_file,0)
        diff_file = np.expand_dims(diff_file,-1)
        
        
        length = file.shape[2] 
        times = file.shape[1]

        if length == 150:
            lengthCutOff = int((150-128)/2)
        elif length == 640:
            lengthCutOff = int((640-512)/2)
        
        orig_img = file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]
        orig_img_diff = diff_file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]

        pred = unet.predict(orig_img)
        pred_diff = unet.predict(orig_img_diff)

        length = orig_img.shape[2]
        times = orig_img.shape[1]

        if plotImages:
            fig,axs=plt.subplots(1,2,figsize=(16,16))
            ax2 = axs[1]
            ax=axs[0]
            im=ax.imshow(pred_diff[0,:,:,0],aspect='auto')
            plt.colorbar(im,ax=ax)
            ax.set_ylabel('t')
            ax.set_title("Segmented Image")
            ax.set_xlabel('x')
            
            loc = plticker.MultipleLocator(base=10.0)
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
            
            if saveImages:
                try:
                    plt.savefig("output/"+fileName+".png")
                    #np.save("output/"+fileName,pred_diff[0,:,:,0])
                except:
                    os.mkdir("output")
                    plt.savefig("output/"+fileName+".png")
                    #np.save("output/"+fileName,pred_diff[0,:,:,0])
                    
        YOLOLabels = manYOLO(pred_diff)
    
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
        
            #Pick out each separate trajectory
            for x1, y1, x2, y2 in detections:
                x1 = np.max([x1,0])
                y1 = np.max([y1,0])
                x2 = np.min([x2,orig_img.shape[2]])
                y2 = np.min([y2,orig_img.shape[1]])
                box_w = x2 - x1
                box_h = y2 - y1    
                yolo_img = orig_img[:,int(y1):int(y2),int(x1):int(x2),:]
                yolo_img_diff = pred_diff[:,int(y1):int(y2),int(x1):int(x2),:]
                
                
                #If the detected trajectory is longer than minTrajLength, predict its intensity & diffusion with the base model and ensemble model.
                if yolo_img.shape[1] > 34 and yolo_img.shape[2] > 34:
                        
                    intensity = resnet_intensity_base(yolo_img,training=False)[0][0][0]
                    diffusion = resnet_diffusion_base(yolo_img_diff,training=False)[0][0][0]**2*57
                                   
                    intensity_ensemble,diffusion_ensemble = predict_function(resnet_ensemble,yolo_img,yolo_img_diff,intensity,diffusion)
                    
                    meanInt = np.mean([intensity,intensity_ensemble])
                    stdInt = np.std([intensity,intensity_ensemble])
                    meanDiff = np.mean([diffusion,diffusion_ensemble])
                    stdDiff = np.std([diffusion,diffusion_ensemble])
                    if stdInt>intensity_ensemble or stdDiff>diffusion_ensemble:
                        print("Ignoring prediction, stdInt = "+str(stdInt)+", meanInt = "+str(meanInt)+", stdDiff = "+str(stdDiff)+", meanDiff = "+str(meanDiff))
                        intensity,diffusion,intensity_ensemble,diffusion_ensemble = 0,0,0,0
                        box_h=0
                else:
                    intensity,diffusion,intensity_ensemble,diffusion_ensemble = 0,0,0,0
                    box_h=0

                if plotImages:
                    # Create a Bounding Box
                    bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, zorder=2,edgecolor="black", facecolor="none")           
                    ax2.text(x1,y1,"I = "+str(np.round(intensity_ensemble,2))+", D = "+str(np.round(diffusion_ensemble,1)),color = "black")
                    ax2.add_patch(bbox)

                #Save results in arrays
                try:
                    intensityArray = np.append(intensityArray,intensity)  
                    diffusionArray = np.append(diffusionArray,diffusion)  
                    intensityEnsembleArray = np.append(intensityEnsembleArray,intensity_ensemble)  
                    diffusionEnsembleArray = np.append(diffusionEnsembleArray,diffusion_ensemble)  
                    meanIntArray = np.append(meanIntArray,meanInt)  
                    meanDiffArray = np.append(meanDiffArray,meanDiff)  
                    countArray = np.append(countArray,box_h)

                except:
                    intensityArray = np.array([intensity])
                    diffusionArray = np.array([diffusion])
                    intensityEnsembleArray = np.array(intensity_ensemble)  
                    diffusionEnsembleArray = np.array(diffusion_ensemble)  
                    meanIntArray = np.array(meanInt)  
                    meanDiffArray = np.array(meanDiff)  
                    countArray = np.array([box_h])

#%% Plot results
plt.close('all')
intensityArray=intensityArray[intensityArray>0]
diffusionArray=diffusionArray[diffusionArray>0]
intensityEnsembleArray= intensityEnsembleArray[intensityEnsembleArray>0]
diffusionEnsembleArray= diffusionEnsembleArray[diffusionEnsembleArray>0]
countArray=countArray[countArray>0]

try:
    os.mkdir("Results")
except:
    pass

from scipy.optimize import curve_fit
from pylab import hist, exp, sqrt, diag, plot, legend
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

data = [[i]*int(j) for i,j in zip(intensityEnsembleArray,countArray)]
data = np.concatenate(data).ravel()
nbr_bins = int(np.max(data)*20)

plt.figure(figsize=(12,6))
y,x,_=hist(data,nbr_bins,alpha=.3,label='data');
x=(x[1:]+x[:-1])/2 # make len(x)==len(y)

iOCRange=[0.75,1,2,5,10,20]
for i in range(0,len(iOCRange)):
    expected_height = len(data)
    expected_peak = iOCRange[i]
    expected_width = expected_peak/100
    
    expected=(expected_peak,expected_width,expected_height)
    print('expected peak = {:.2f}'.format(expected_peak))
    print('expected width = {:.2f}'.format(expected_width))
    params,cov=curve_fit(gauss,x,y,expected)
    sigma=sqrt(diag(cov))
    plot(x,gauss(x,*params),lw=2,label='Gaussian fit, iOC='+str(iOCRange[i]) + "$\cdot 10^{-4}~nm$")
    plt.xlabel("iOC ($\cdot 10^{-4} nm$)")
    plt.ylabel("#Frames")
    peak = params[0]
    sigma = abs(params[1]) # sometimes the width is outputted as a negative number...
    
    legend(fontsize=14)
    plt.title("iOC Histogram")
    
    print('peak = {:.2f}'.format(peak))
    print('width = {:.2f}'.format(sigma))

plt.savefig("Results/iOCHistogram")
plt.show()

#data = np.copy(diffusionEnsembleArray)
data = [[i]*int(j) for i,j in zip(diffusionEnsembleArray,countArray)]
data = np.concatenate(data).ravel()
nbr_bins = int(np.max(data)*20)

plt.figure(figsize=(12,6))
y,x,_=hist(data,nbr_bins,alpha=.3,label='data');
x=(x[1:]+x[:-1])/2 # make len(x)==len(y)

DRange=[10,20,50]
for i in range(0,len(DRange)):
    expected_height = len(data)
    expected_peak = DRange[i]
    expected_width = expected_peak/10
    
    expected=(expected_peak,expected_width,expected_height)
    print('expected peak = {:.2f}'.format(expected_peak))
    print('expected width = {:.2f}'.format(expected_width))
    params,cov=curve_fit(gauss,x,y,expected)
    sigma=sqrt(diag(cov))
    plot(x,gauss(x,*params),lw=2,label='Gaussian fit, D='+str(DRange[i])+"$\mu m^2/s$")
    plt.xlabel("D ($\mu m^2/s$)")
    plt.ylabel("#Frames")
    peak = params[0]
    sigma = abs(params[1]) # sometimes the width is outputted as a negative number...
    
    legend(fontsize=14)
    plt.xlim(0,75)
    
    plt.title("D Histogram")
    
    print('peak = {:.2f}'.format(peak))
    print('width = {:.2f}'.format(sigma))

plt.savefig("Results/DHistogram")
plt.show()

points=list(set(zip(intensityEnsembleArray,diffusionEnsembleArray))) 
plot_x=[i[0] for i in points]
plot_y=[i[1] for i in points]
count=np.array(countArray)
plt.figure(figsize=(12,12))
plt.scatter(plot_x,plot_y,color="blue",alpha=count/np.max(count))
plt.xlabel("iOC ($\cdot 10^{-4} nm$)")
plt.ylabel("D ($\mu m^2/s$)")
plt.yscale('log')
plt.title("iOC vs D scatter-plot")
plt.ylim(0,100)
plt.gca().invert_yaxis()
plt.savefig("Results/Scatter_plot")
plt.show()


