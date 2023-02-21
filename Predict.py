
import matplotlib.ticker as plticker
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from utils.YOLO import manYOLO,manYOLOSplit, ConvertYOLOLabelsToCoord, ConvertCoordToYOLOLabels
from utils.funcs import remove_stuck_particle_2

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

resnet_diffusion_path='Network-weights/resnet-diffusion-09062021-0734012048x128_combinedLoss.h5' #Normal diffusion model
resnet_diffusion_path_EV='Network-weights/resnet-diffusion-21072021-054520512x128_EV_combinedLoss.h5' #EV diff model

resnet_intensity_path='Network-weights/resnet-D0.1-1.15 I0.01-25 128x128_loss_5.546048.h5' #Normal int model
resnet_intensity_path_EV = "Network-weights/resnet-D0.05-1.15 I0.5-1000 512x512_loss_104.02502.h5" #EV int model

resnet_intensity=reload_resnet()
resnet_diffusion=reload_resnet()

#If doing exosome analysis, load ..._EV models here instead. 
unet = tf.keras.models.load_model(unet_path,compile=False)
resnet_intensity.load_weights(resnet_intensity_path)
resnet_diffusion.load_weights(resnet_diffusion_path)


#%% Enter user variables

#Path to 
mainDataPath='Data/...'
#Experimental or simulated data?
experimental_data=True
#Plot data?
plot=True
#Save images?
savePredDiff = False
#Minimal accepted trajectory length (FCNN downsamples 5 times -> minimal accepted length is 2^5=32)
minImgArea = 32
#%%    

try:
    del intensityArray
    del diffusionArray
    del countarray
except:
    pass

measurements = os.listdir(mainDataPath)
print("Measurements in file: " +measurements)
#measurements = [measurements[0]]
for meas in measurements:
    print("Running measurement: "+meas)
    dataPath = mainDataPath+"/"+meas
    files = os.listdir(dataPath + "/intensity/")

    for fileName in files:
        
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
        img_size=256
        timesLimit = times % img_size
        if length == 150:
            lengthCutOff = (150-128)/2
        elif length == 640:
            lengthCutOff = (640-512)/2
        
        
        #Data come in all sorts of weird shapes and size, here we manage it..
        if not experimental_data:
            orig_img = file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]
            orig_img_diff = diff_file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]
        elif timesLimit > 0:
            if file.shape[1] >= 8192+904:
                orig_img = file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]
                orig_img_diff = diff_file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]
            else:
                orig_img = file[:,0:-int(timesLimit),lengthCutOff:-lengthCutOff,:]
                orig_img_diff = diff_file[:,0:-int(timesLimit),lengthCutOff:-lengthCutOff,:]
        else:
            orig_img = file[:,:,lengthCutOff:-lengthCutOff,:]
            orig_img_diff = diff_file[:,:,lengthCutOff:-lengthCutOff,:]

        pred = unet.predict(orig_img)
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

            
            # try:
            #     plt.savefig("output/"+fileName+".png")
            #     np.save("output/"+fileName,pred_diff[0,:,:,0])
            # except:
            #     os.mkdir("output")
            #     plt.savefig("output/"+fileName+".png")
            #     np.save("output/"+fileName,pred_diff[0,:,:,0])
            

        YOLOLabels = manYOLO(pred_diff,trajTreshold=16)

        #YOLOLabels = manYOLOSplit(pred_diff)
    
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
                box_h = y2 - y1    

                #Resnet-intensity predictions
                yolo_img = orig_img[:,int(y1):int(y2),int(x1):int(x2),:]
                yolo_img_diff = pred_diff[:,int(y1):int(y2),int(x1):int(x2),:]
                
                if yolo_img.shape[1] >= 32 and yolo_img.shape[2] >= 32 and yolo_img.shape[2]*yolo_img.shape[1] > minImgArea:
                    

                    M = 10
                    nbr_its = 5

                    if True:
                        yolo_img_diff = remove_stuck_particle_2(yolo_img_diff,M,nbr_its)
                        

                    intensity = resnet_intensity(yolo_img,training=False)[0][0][0]
                    diffusion = resnet_diffusion(yolo_img_diff,training=False)[0][0][0]**2*57

                else:
                    intensity = 0
                    diffusion = 0
                    box_h=1

                if plot:
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, zorder=2,edgecolor="black", facecolor="none")

                   # if int(cls_pred) == 0:
                    ax2.text(x1,y1,"I = "+str(np.round(intensity,2))+", D = "+str(np.round(diffusion,1)),color = "black")
                    ax2.add_patch(bbox)



                try:
                    intensityArray = np.append(intensityArray,intensity)  
                    diffusionArray = np.append(diffusionArray,diffusion)  
                    countArray = np.append(countArray,box_h)

                except:
                    intensityArray = np.array([intensity])
                    diffusionArray = np.array([diffusion])
                    countArray = np.array([box_h])


