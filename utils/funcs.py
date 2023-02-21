import numpy as np
from scipy.signal import convolve2d
import scipy
import sys
sys.path.append("deeptrack/")
from deeptrack.models import resnetcnn
from tensorflow.keras import models

def remove_stuck_particle_2(original_img,M,nbr_its):
    original_img = original_img[0,:,:,0]
    #plt.figure()
    #plt.imshow(original_img,aspect='auto')
    for i in range(0,nbr_its):

        conv_img = original_img - convolve2d(original_img,np.ones((M,1))/M,mode='same',boundary='symm',fillvalue=1)
        
        img = original_img * (1-conv_img)
        try:
            img /= np.max(img)
        except:
            return original_img
        
        img[img<0.99] = 0
        img[img>0] = 1

        identifiedStuckTraj = np.sum(img,0)
        img[:,identifiedStuckTraj<2] = 0
        
        binary_img = scipy.ndimage.morphology.binary_dilation(img,structure=np.ones((M,1)))

        idcs = np.sum(binary_img,axis=0)==0
        cut_img = original_img[:,idcs]
        original_img = np.copy(cut_img)

    return np.expand_dims(original_img,(0,-1))

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



def load_all_models(): 
            
        resnet_path_075_1 = "Network-weights/resnet-D0.01-1 I0.0-1.5 512x128_loss_0.1030596.h5"
        resnet_path_075 = "Network-weights/resnet-D0.05-1.15 I0.01-0.99 2048x128_loss_0.002396633.h5"
        resnet_path_1 = "Network-weights/resnet-D0.05-1.15 I0.8-1.9 2048x128_loss_0.011120575.h5"
        resnet_path_2 = "Network-weights/resnet-D0.05-1.15 I1.5-2.5 512x128_loss_0.0037260056.h5"
        resnet_path_5 = "Network-weights/resnet-D0.01-1 I2.5-7.5 512x128_loss_0.03088841.h5"
        resnet_path_10 = "Network-weights/resnet-D0.05-1.15 I5.1-15 512x128_loss_0.025085581.h5"
        resnet_path_20 = "Network-weights/resnet-D0.01-1 I15-25 512x128_loss_0.13881019.h5"
        
        resnet_diff_path_10 = "Network-weights/resnet-diffusion-D0.57-14.25 I0.01-30 2048x128_loss_0.03129236.h5"
        resnet_diff_path_20 = "Network-weights/resnet-diffusion-D11.54-36.48 I0-50 512x128_loss_0.13375391.h5"
        resnet_diff_path_50 = "Network-weights/resnet-diffusion-D27.93-75.38 I0.01-30 2048x128_loss_0.14572684.h5"
        
        resnet_intensity_075_1 = reload_resnet()
        resnet_intensity_075 = reload_resnet()
        resnet_intensity_1 = reload_resnet()
        resnet_intensity_2 = reload_resnet()
        resnet_intensity_5 = reload_resnet()
        resnet_intensity_10 = reload_resnet()
        resnet_intensity_20 = reload_resnet()
        
        resnet_diffusion_10= reload_resnet()
        resnet_diffusion_20 = reload_resnet()
        resnet_diffusion_50 = reload_resnet()

        resnet_intensity_075_1.load_weights(resnet_path_075_1)
        resnet_intensity_075.load_weights(resnet_path_075)
        resnet_intensity_1.load_weights(resnet_path_1)
        resnet_intensity_2.load_weights(resnet_path_2)
        resnet_intensity_5.load_weights(resnet_path_5)
        resnet_intensity_10.load_weights(resnet_path_10)
        resnet_intensity_20.load_weights(resnet_path_20)
        
        resnet_diffusion_10.load_weights(resnet_diff_path_10)
        resnet_diffusion_20.load_weights(resnet_diff_path_20)
        resnet_diffusion_50.load_weights(resnet_diff_path_50)
        
        iOCRange=[0.75,0.88,1,2,5,10,20]
        DRange=[10,20,50]
        
        resnetiOC = [resnet_intensity_075,resnet_intensity_075_1,resnet_intensity_1,resnet_intensity_2,resnet_intensity_5,resnet_intensity_10,resnet_intensity_20]
        resnetD = [resnet_diffusion_10,resnet_diffusion_20,resnet_diffusion_50]
        
        def get_models(models,idx,prop,model):
                
            models["idx"]=np.append( models["idx"],idx)
            models["prop"]=np.append( models["prop"],prop)
            models["model"]=np.append( models["model"],model)
            
            return models
        
        iOCModels = {
            "idx": [],
            "prop": [],
            "model": [],
            }
        diffModels = {
            "idx": [],
            "prop": [],
            "model": [],
            }
        for i in range(len(iOCRange)):
            iOCModels = get_models(iOCModels,i, iOCRange[i],resnetiOC[i])
            
        for i in range(len(DRange)):
            diffModels = get_models(diffModels,i, DRange[i],resnetD[i])

        
        return iOCModels,diffModels
    
def predict_function(ensemble,img,img_diff,intensity,diffusion):
    
    I=np.abs(ensemble[0]["prop"]-intensity)
    indModel = np.where(I==np.min(I))[0][0]
    model = ensemble[0]["model"][indModel]
    intensity = model.predict(img)[0][0][0]
    
    if diffusion < 15:
        model = ensemble[1]["model"][0]
    elif diffusion < 30:
        model = ensemble[1]["model"][1]
    else:
        model = ensemble[1]["model"][2]
        
    diffusion = model.predict(img_diff)[0][0][0]**2*57
    return intensity,diffusion