#%% Imports
import sys
sys.path.append("..") #Adds the module to path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as IO
import os
import tqdm as tqdm
import skimage.measure
from scipy.signal import convolve2d
import pickle
from scipy.ndimage import median_filter


#%  Define preprocessing functions

# Doesn't scale vs std of background (used for intensity predictions)
def intensity_preprocess(A):

    B=(A-np.mean(A,axis=0))/np.mean(A,axis=0) # Remove the
    ono=np.ones((200,1))
    ono=ono/np.sum(ono)
    
    # remove local mean \pm 100 pixels
    B-=convolve2d(B,ono,mode="same")
    B-=convolve2d(B,np.transpose(ono),mode="same")
    
    B-=np.expand_dims(np.mean(B,axis=0),axis=0)
    
    B_pre_blockreduce = np.copy(B)
    noise_std = np.std(B,axis=0)
    B=1000*skimage.measure.block_reduce(B,(1,4),np.mean) #We multiply by a large valeu to accentuate pixel value differences (higher values mean potentially better contrast LOD/accuracy?)
    return B, B_pre_blockreduce, noise_std

def diffusion_preprocess_new(B_pre_blockreduce, noise_std):
    # A is preprocessed with intensity
    B2 = np.copy(B_pre_blockreduce)
    B2=B2/noise_std
    B2=skimage.measure.block_reduce(B2,(1,4),np.mean)
    return B2

# Scales image vs background (used for segmentation and diffusion predictions) /7 jan
def diffusion_preprocess_old(A):
    # A is preprocessed with intensity
    B2=(A-np.mean(A,axis=0))/np.mean(A,axis=0)

    ono=np.ones((200,1))
    ono=ono/np.sum(ono)

    B2-=convolve2d(B2,ono,mode="same")
    B2-=convolve2d(B2,np.transpose(ono),mode="same")
    B2-=np.expand_dims(np.mean(B2,axis=0),axis=0)
    a=np.std(B2,axis=0)
    B2=B2/a
    B2=skimage.measure.block_reduce(B2,(1,4),np.mean)
    return B2,a

def stabilize_Barbora(im):
    length = np.size(im,1)
    x=np.linspace(-1,1,length)
    kx=np.fft.fftshift(np.pi*np.linspace(-1,1,length))/(x[1]-x[0])
    dx=0
    imm=im[...]
    for j in range(10):
    
        dx0=np.array([np.sum(x*imm[i,:])/np.sum(imm[i,:]) for i in range(im.shape[0])])
        imm=np.array([np.real(np.fft.ifft(np.fft.fft(imm[i,:])*np.exp(1j*kx*(dx0[i]-dx0[0])))) for i in range(im.shape[0])])
        dx+=np.array(dx0)
    
    return imm

def intensity_preprocess_lambdaDNA(A):
    times = np.where(A.shape==np.max(A.shape))[0][0]
    if times == 1:
        A=np.transpose(A)
        
    B=(A-np.mean(A,axis=0))/np.mean(A,axis=0) # Remove the
    B = B/np.std(B,axis=0)
    return B

def preprocess_basic(A):
    times = np.where(A.shape==np.max(A.shape))[0][0]
    if times == 1:
        A=np.transpose(A)
        
    B=(A-np.mean(A,axis=0))/np.mean(A,axis=0) # Remove the
    B = B/np.std(B,axis=0)
    return B
        
def load_pkl(filepath):
    load_file = open(filepath, "rb")
    load_file = pickle.load(load_file)
    print("Loaded file " + filepath)
    return load_file


        
#%% Preprocess and save
plt.close('all')

#User-defined variables

#Length-wise downscaling factor. Default=4
downscaling_factor = 4

#Save files? Default=True
save_files = True
#Plot processed data? Default=False
plot = False
#Save images of processed data alongside it? Default=False
save_pictures = False

#Split images where camera errors are detected. Default=True
split_images = True

#Images to be processed
img_path="Data/Demo Data/"
#Path to save images
main_save_path ='Data/Preprocessed Sample Data/'

folders = os.listdir(img_path)
folders = [folder for folder in folders if "." not in folder]
for folder in tqdm.tqdm(folders):
    
    load_path = img_path + folder
    save_path = main_save_path + folder
    diffusion_save_path = save_path + '/diffusion/'
    intensity_save_path = save_path + '/intensity/'
    pictures_save_path = save_path + '/pictures/'
    
    try:
       os.makedirs(diffusion_save_path, exist_ok=True)
    except:
        print()
        print('Directory:' + diffusion_save_path + ' already exists, passing.')
        continue
    try:
       os.makedirs(intensity_save_path, exist_ok=True)
    except:
        print()
        print('Directory:' + intensity_save_path + ' already exists, passing.')
        continue
    try:
       os.makedirs(pictures_save_path, exist_ok=True)
    except:
        print()
        print('Directory:' + pictures_save_path + ' already exists, passing.')
        continue

    
    files = os.listdir(load_path)
    existingFiles = os.listdir(intensity_save_path)
    files = [file for file in files if file.endswith('_M.mat')] #Change this for exosomes
    print('len(files):' + str(len(files)))
    frame_nbr = ''

    for filename in tqdm.tqdm(files):
        
        if filename+".npy" in existingFiles:
            continue
                
        ExpData = IO.loadmat(load_path + '/' + filename)
        

        
        try:
            full_data = ExpData["data"]["Im"][0][0]
            times = np.where(full_data.shape==np.max(full_data.shape))[0][0]
            if times == 1:
                    full_data=np.transpose(full_data)
        except:
            print("Failed: "+load_path+'/'+filename)
            continue
        
        time = ExpData["data"]["time"][0][0][0]
        pos = ExpData["data"]["Yum"][0][0][0]

        ds = pos[1]-pos[0]
        #print(ds)
        dt = time[1]-time[0] # seconds per frame
        #print(dt)
        s1 = 12
        s2 = 2.3
        fontsize = 11
        #plt.figure(figsize=(s1,s2))
        
        
        x1 = 0
        x2 = full_data.shape[0]*dt
        y1 = 0
        y2 = full_data.shape[1]*ds
       
        # Loop through image to check if camera has blank shots
        idcs = []
        for j,timestep in enumerate(full_data[:,0]):
            if np.all(full_data[j,:] == np.zeros(full_data[j,:].shape)):
                idcs.append(j)
        camera_has_blank_shots = len(idcs) > 0
        #camera_has_blank_shots = 0
        if camera_has_blank_shots:
                    
            prev_idx = 0
            shift = 50
            for j,idx in enumerate(idcs):
                data = full_data[prev_idx+shift:idx-shift]
                prev_idx = idx

                
                Bsm_iOC, Bsm_pre_blockreduce, noise_std = intensity_preprocess(data)
                Bsm_D = diffusion_preprocess_new(Bsm_pre_blockreduce,noise_std)  

                # Plot diffusion img
                if plot:
                    plt.figure(figsize=(16,2))
                    i1 = 256
                    plt.imshow(Bsm_D[11:-11].T,aspect='auto')#,vmin=-1,vmax=1)
                    plt.title(filename)
                    plt.colorbar()
                
                if idx != len(files):
                    frame_nbr = str(idx)
                else:
                    frame_nbr = ''
                if save_files:
                    np.save(diffusion_save_path + filename + frame_nbr, Bsm_D)
                    np.save(intensity_save_path + filename + frame_nbr, Bsm_iOC)
                    print(' Files saved.')
                if save_pictures:
                    plt.savefig(pictures_save_path+filename[:-4]+ frame_nbr +'.png')
                    #plt.close('all')
                save_csv_matrix = 0
                if save_csv_matrix:
                    np.savetxt('{}_{}_{}.csv'.format(measurement,file_to_search_for,frame_nbr),Bsm_D,delimiter=',')
        else:    
            data = np.copy(full_data)

            Bsm_iOC, Bsm_pre_blockreduce, noise_std = intensity_preprocess(data)
            Bsm_D = diffusion_preprocess_new(Bsm_pre_blockreduce,noise_std)  
                
                
           # BSm_s = stabilize_Barbora(data)
            if plot:
                try:
                    plt.figure(figsize=(s1,s2))
                    plt.imshow(Bsm_D[11:-11].T,aspect='auto', extent=[x1,x2,y1,y2])#,vmin=-1,vmax=1)
                    plt.title('Preprocessed image for diffusion',fontsize=fontsize)
                    plt.xlabel(r'Time (s)',fontsize=fontsize)
                    plt.ylabel(r'Position ($\mu$m)',fontsize=fontsize)
                    plt.colorbar()   
                    plt.tight_layout()
                    plt.savefig('diffusion.pdf')
                except:
                    pass
                plt.figure(figsize=(s1,s2))
                plt.imshow(Bsm_iOC[11:-11].T,aspect='auto', extent=[x1,x2,y1,y2])#,vmin=-1,vmax=1)
                plt.title('Preprocessed image for optical contrast',fontsize=fontsize)
                plt.xlabel(r'Time (s)',fontsize=fontsize)
                plt.ylabel(r'Position ($\mu$m)',fontsize=fontsize)
                plt.colorbar()   
                plt.tight_layout()
                plt.savefig('intensity.pdf')
            if save_files:
                try: #If no diffusion processing exists, pass. E.g. for lambda-DNA
                    np.save(diffusion_save_path + filename + frame_nbr, Bsm_D)
                except:
                    pass
                np.save(intensity_save_path + filename + frame_nbr, Bsm_iOC)
                try:
                    traj = ExpData["data"]["responce"][0][0]
                    np.save(save_traj_path+filename+".npy",traj)
                    print('Saving trajectory')
                except:
                    pass
                print(' Files saved.')
            
            if save_pictures:
                plt.savefig(pictures_save_path+filename[:-4]+ '.png')
                plt.close('all')
                
  
            
                
#%% #extract ground truth trajectories from simulated data
simPath ="Z:/NSM/simulated_tests/simulated_diffusing_molecule/test/" #"C:/Users/ccx55/temp/velocity0_distance20_timesteps10000/"
#simPath=#../Data/simulated_exosome/"
savePath="../Data/Preprocessed test Simulated Data Ground Truth/"
simFiles = os.listdir(simPath)
for file in simFiles:
    if not file.endswith("Ds.mat"):
        ExpData = IO.loadmat(simPath+file)
        try:
            traj = ExpData["data"]["responce"][0][0]
        except:
            traj = ExpData["simulated"]["responce"][0][0]
        np.save(savePath+file+".npy",traj)

