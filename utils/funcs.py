import numpy as np
from scipy.signal import convolve2d
import scipy
import sys
sys.path.append("deeptrack/")
from deeptrack.models import resnetcnn
from tensorflow.keras import models
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit



def remove_stuck_particle_2(original_img,M,nbr_its):
    """  This function removes stuck particles from the image
    Input: 
    original_img: the original image
    M: the number of rows
    nbr_its: number of iterations for the algorithm
    Output:
    cut_img: the image after removing stuck particles 
    """
    
    # M is the size of the kernel, nbr_its is the number of iterations
    # the code performs the following operations:
    #   1. convolve the original image with a kernel of size M
    #   2. subtract the convolved image from the original image
    #   3. multiply the original image by the difference image
    #   4. apply a threshold to the resulting image
    #   5. perform a morphological dilation on the thresholded image
    #   6. remove any columns in the original image that are all 0s
    #   7. repeat for nbr_its iterations

    original_img = original_img[0,:,:,0]

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
    
    # Find the model that best matches the input intensity
    I=np.abs(ensemble[0]["prop"]-intensity)
    indModel = np.where(I==np.min(I))[0][0]
    model = ensemble[0]["model"][indModel]
    intensity = model.predict(img)[0][0][0]
    
    # Find the model that best matches the input diffusion
    if diffusion < 15:
        model = ensemble[1]["model"][0]
    elif diffusion < 30:
        model = ensemble[1]["model"][1]
    else:
        model = ensemble[1]["model"][2]
        
    diffusion = model.predict(img_diff)[0][0][0]**2*57
    return intensity,diffusion

def manTrack(diffImg, keepTraj=16, threshold=0.05, trajTreshold=128, dist=32):
    """
    Track object trajectories in a sequence of difference images.

    Parameters:
    -----------
    diffImg : numpy.ndarray
        A sequence of difference images.
    keepTraj : int, optional (default=16)
        Minimum length of accepted trajectory.
    threshold : float, optional (default=0.05)
        Minimum height of local maxima.
    trajTreshold : int, optional (default=128)
        Minimum distance from trajectory to point to consider point in trajectory.
    dist : int, optional (default=32)
        Minimum distance between trajectories.

    Returns:
    --------
    trajectories : list of list of tuples
        A list of trajectories, where each trajectory is a list of (frame, position) tuples.
    num_trajectories : int
        The number of tracked trajectories.
    frames : dict of numpy.ndarray
        A dictionary of frames with local maxima.
    """
    frames = {}

    # Find local maxima in each frame
    for f in range(diffImg.shape[0]):
        frame = diffImg[f, :]
        local_maxima, _ = find_peaks(frame, height=threshold, distance=dist)
        if len(local_maxima) > 0:
            frames[f] = local_maxima

    # Initialize the trajectories
    trajectories = {}
    trajectory_id = 0

    for frame, local_maxima in frames.items():
        current_trajectories = len(trajectories)

        # Compute the distance between local maxima and existing trajectories
        distances = np.full((current_trajectories, len(local_maxima)), np.inf)
        for t, traj in trajectories.items():
            last_frame, last_position = traj[-1]
            traj_distances = np.sqrt(np.sum((local_maxima[:, None] - last_position) ** 2, axis=1))
            if frame - last_frame < trajTreshold:
                distances[t, :] = traj_distances

        # Associate local maxima with existing trajectories or create new trajectories
        not_taken_trajectories = np.arange(len(local_maxima))
        not_used_trajectories = np.arange(current_trajectories)
        for i, local_max in enumerate(local_maxima):
            if len(distances[:,i])==0:
                continue
            t = np.argmin(distances[:, i])
            if distances[t, i] < np.inf and t in not_used_trajectories and np.abs(local_max - trajectories[t][-1][1]) < dist:
                not_taken_trajectories = np.delete(not_taken_trajectories, np.where(not_taken_trajectories == i))
                not_used_trajectories = np.delete(not_used_trajectories, np.where(not_used_trajectories == t))
                trajectories[t].append((frame, local_max))

        for i in not_taken_trajectories:
            trajectories[trajectory_id] = [(frame, local_maxima[i])]
            trajectory_id += 1

    # Filter out short trajectories
    trajectories = [t for t in trajectories.values() if len(t) > keepTraj]
    num_trajectories = len(trajectories)

    return trajectories, num_trajectories, frames


def CalculateiOCandD(trajectories, currentTrajectories, image=None, plot=False):
    """
    This function calculates the intensity and diffusivity of particles from input trajectories and an optional image.

    Args:
        trajectories (list of lists): A list of particle trajectories, where each trajectory is a list of (time, x, y) tuples.
        current_trajectories (int): The number of trajectories in the list.
        image (numpy array): An image to calculate iOC on.
        plot (optinal, boolean) plot the particle positions on aforementioned image.

    Returns:
        fullIntensity (numpy array): An array of intensities for each particle in the trajectories.
        fullDiffusivity (numpy array): An array of diffusivities for each particle in the trajectories.
    """
    
      # Define a Gaussian function to use for curve fitting
    def gauss(x, A, b):
        y = A * np.exp(-x ** 2 / b ** 2)
        return y
    # Define a list of colors to plot each trajectory with
    colourList = np.tile(plt.rcParams['axes.prop_cycle'].by_key()['color'], 5)
    
    # Initialize arrays for diffusivity and intensity
    D = np.zeros(currentTrajectories)
    I = np.zeros(currentTrajectories)
    
    # Get the length of each trajectory
    trajectoryLengths = [len(trajectories[i]) for i in range(0, currentTrajectories)]

    # If an image is provided, plot it
    if plot:
        plt.figure()
        plt.imshow(image, aspect='auto')

    # Loop through each trajectory and calculate diffusivity and intensity
    for t in range(currentTrajectories):
        time = np.array([trajectories[t][k][0] for k in range(1, len(trajectories[t]))])
        x = [trajectories[t][k][1] for k in range(1, len(trajectories[t]))]

        if plot:
            plt.scatter(x, time, s=8, c=colourList[t])
            plt.plot(x, time, c=colourList[t])
            
        # Calculate intensity using curve fitting
        xtomu = 0.029545454545454545 * 4  # assuming a downsampling factor of 4
        xdata = np.arange(-10, 10) * xtomu
        intensity = np.zeros(len(trajectories[t]))

        for i in range(len(trajectories[t])):
            if trajectories[t][i][1] < 10:
                ydata = None
            elif trajectories[t][i][1] > image.shape[1] - 10:
                ydata = None
            else:
                ydata = image[trajectories[t][i][0], (trajectories[t][i][1]) - 10:(trajectories[t][i][1] + 10)]

            try:
                parameters, covariance = curve_fit(gauss, xdata, -ydata)
                intensity[i] = np.sqrt(np.pi) * parameters[0] * parameters[1]
            except:
                pass

        I[t] = np.median(intensity[intensity > 0])
        dt=0.0049893
        # compute diffusion coefficient
        D[t] = np.mean([np.abs(trajectories[t][i][1] * xtomu - trajectories[t][i + 1][1] * xtomu) ** 2
                        for i in range(1, len(trajectories[t]) - 1)]) / 2 / dt
        D[t] += np.mean([np.abs((trajectories[t][i + 1][1] - trajectories[t][i + 2][1]) *
                                (trajectories[t][i][1] - trajectories[t][i + 1][1])) * xtomu ** 2
                         for i in range(1, len(trajectories[t]) - 2)]) / dt
        D[t] /= 2

    keepInds = np.where(~np.isnan(I))[0]
    I = I[keepInds]
    D = D[keepInds]
    trajectoryLengths = np.array(trajectoryLengths)[keepInds]

    return I, D, trajectoryLengths
