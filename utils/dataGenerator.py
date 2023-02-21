
from deeptrack.features import Feature
import tensorflow as tf
from tensorflow import keras
import numpy as np
import skimage.measure
from scipy.signal import convolve
from scipy.ndimage import median_filter
K=keras.backend
#import cupy as np
median_process = 0

size = 256
length = 4*128
L_reduction_factor = 4
reduced_length = int(length/L_reduction_factor)

times = 2048
T_reduction_factor = 1
reduced_times = int(times/T_reduction_factor)

class get_diffusion_and_intensity(Feature):
    
    def __init__(self, vel=0, D=[], I=[], s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

        self.D1 = D[0]
        self.D2 = D[1]

        self.I1 = I[0]
        self.I2 = I[1]

    def get(self, image, vel, D, I,s, **kwargs):
        ### Set parameters here ###
        length=image.shape[1]
        times=image.shape[0]
        # s is the Gaussian width of the simulated particles
        image.append({"s":s})
        
        D = (self.D1 + self.D2*np.random.rand())

        I = (self.I1+self.I2*np.random.rand())
        
        pixelSize = 0.0295
        
        image.append({"D":D})
        image.append({"I_initial":s*I*np.sqrt(2*np.pi)*length/2*pixelSize}) ###changed iOC here!!!!!
        image.append({"I":s*I*np.sqrt(2*np.pi)*length/2*pixelSize}) 

        return image
    
class get_trajectory(Feature):

    def __init__(self, vel=0, D=0.1, I=0.1,s=0.05, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        s = image.get_property("s")
        D = image.get_property("D")/10
        I = image.get_property("I_initial")/10000
        
        length = image.shape[1]
        times = image.shape[0]
        x = np.linspace(-1,1,length)
        t = np.linspace(-1,1,times)
        X,Y = np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        x0 = -1+2*np.random.rand()
        x0 += np.cumsum(vel+D*np.random.randn(times))
        v1 = np.transpose(I*f2(1,x0,s,0,Y))
        image[...,0] *= (1-v1)
        image[...,1] += np.transpose(f2(1,x0,s,0,Y))
        
        
        return image
    
class gen_noise(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0,biglam=0,bgnoiseCval=0,bgnoise=0,bigx0=0,sinus_noise_amplitude=0,freq=0,**kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA,biglam=biglam,bgnoiseCval=bgnoiseCval,bgnoise=bgnoise,bigx0=bigx0,sinus_noise_amplitude=sinus_noise_amplitude,freq=0, **kwargs
        )
        
   # @njit
    def get(self, image, noise_lev, dX, dA,biglam,bgnoiseCval,bgnoise,bigx0,sinus_noise_amplitude,freq, **kwargs):
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        bgnoise*=np.random.randn(length)

        tempcorr=3*np.random.rand()
        dAmp=dA
        shiftval=dX*np.random.randn()
        dx=0
        dx2=0
        dAmp0=0
        
        bg0=f2(1,bigx0,biglam,0,x)
        image.append({"bg0":bg0})
        ll=(np.pi-.05)
        for j in range(times):
            dx=(sinus_noise_amplitude*np.random.randn()+np.sin(freq*j))*dX
            
            bgnoiseC=f2(1,0,bgnoiseCval,dx,x)
            bgnoiseC/=np.sum(bgnoiseC)
            bg=f2(1,bigx0+dx,biglam,0,x)*(1+convolve(bgnoise,bgnoiseC,mode="same"))
            dAmp0=dA*np.random.randn()
            bg*=(1+dAmp0)
            image[j,:,0]=bg*(1+noise_lev*np.random.randn(length))+.4*noise_lev*np.random.randn(length)
        
        return image
    
    
class post_process_int(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA, **kwargs
        )

    def get(self, image, **kwargs):
        from scipy.signal import convolve2d
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        G= lambda a,b,x0,s,x: a*np.exp(-(x-x0)**2/s**2)+b
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b                         

        #bg0 = image.get_property("bg0") 
        #image[:,:,0]/=bg0
        #image[:,:,0]=(image[...,0]-np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0))/np.mean(image[...,0],axis=0)
        
        image[:,:,0] = (image[:,:,0]-np.mean(image[:,:,0],axis=0))/np.mean(image[:,:,0],axis=0)

        # Perform same preprocessing as done on experimental images
        if not median_process:
            ono=np.ones((200,1))
            ono=ono/np.sum(ono)
            image[:,:,0]-=convolve2d(image[:,:,0],ono,mode="same")
            image[:,:,0]-=convolve2d(image[:,:,0],np.transpose(ono),mode="same")
        else:
            image[:,:,0] -=median_filter(image[:,:,0],size=(200,1))
            image[:,:,0] -=median_filter(image[:,:,0],size=(1,200))
        
        image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)
        image[:,:,0]*=1000 
        
        return image


    
class input_array(Feature):
    __distributed__ = False
    def get(self,image,**kwargs):
        image=np.zeros((times,length,2))
        return image

        
def batch_function_int(image):
    img = image[...,:1]
    img = skimage.measure.block_reduce(img,(T_reduction_factor,L_reduction_factor,1),np.mean)
    return img 

def label_function_int(image):
    D = image.get_property("D")
    I = image.get_property("I")
    length=image.shape[1]
    times=image.shape[0]

    if np.sum(image[...,1]) < 2:
        D = 0
        I = 0
    t_final = int(times/32)
    L_final = int(length/32)
    time_reduction = int(image.shape[0]/t_final)
    len_reduction = int(image.shape[1]/L_final)
    
    mask = skimage.measure.block_reduce(image[...,1:2], (time_reduction,len_reduction,1), np.mean)

    labels = np.ones((t_final,L_final,2))
    labels[...,0] *= I
    labels[...,1:] = mask
    return labels

class init_particle_counter(Feature):
    
    def __init__(self, vel=0, D=0, I=0, s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        
        # Init particle counter
        nbr_particles = 0
        image.append({"nbr_particles":nbr_particles})
        
        return image
    
    
class post_process_diff(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA, **kwargs
        )

    def get(self, image, **kwargs):
        from scipy.signal import convolve2d
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        
        # New preprocess 7 Mars
        #image[:,:,0]=(image[...,0]-np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0))/np.mean(image[...,0],axis=0)
        
        image[:,:,0] = (image[:,:,0]-np.mean(image[:,:,0],axis=0))/np.mean(image[:,:,0],axis=0)
        #Perform same preprocessing as done on experimental images
        ono=np.ones((200,1))
        ono[0:80]=1
        ono[120:]=1
        ono=ono/np.sum(ono)
        image[:,:,0]-=convolve2d(image[:,:,0],ono,mode="same")
        image[:,:,0]-=convolve2d(image[:,:,0],np.transpose(ono),mode="same")
        
        image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)
        a=np.std(image[...,0],axis=0)
        image[:,:,0]/=a
        try:
            image.properties["I"]/=a
        except:
            pass      
        
        return image    
    
        
def batch_function_diff(image):
    img = image[...,:1]
    img = skimage.measure.block_reduce(img,(T_reduction_factor,L_reduction_factor,1),np.mean)
    GAN_output = unet.predict(np.expand_dims(img,axis=0))
    segmentation = np.copy(GAN_output[0])
    segmentation[segmentation < 0] = 0
    return segmentation

def label_function_diff(image):
    D = image.get_property("D")
    I = image.get_property("I")
    length=image.shape[1]
    times=image.shape[0]

    if np.sum(image[...,1]) < 2:
        D = 0
        I = 0
    t_final = int(times/32)
    L_final = int(length/32)

    time_reduction = int(image.shape[0]/t_final)
    len_reduction = int(image.shape[1]/L_final)
    
    mask = skimage.measure.block_reduce(image[...,1:2], (time_reduction,len_reduction,1), np.mean)
    labels = np.ones((t_final,L_final,2))
    labels[...,0] *= D#**2*57
    labels[...,1:] = mask
    
    return labels


mae = tf.keras.losses.MeanAbsoluteError()
huber = tf.keras.losses.Huber()
mse = tf.keras.losses.MeanSquaredError()
def diffusion_loss(T,P):
    D_true = T[:,0,0,0] # Diffusion
    D_pred = P
    
    return mae(D_true,D_pred) 

def diffusion_loss_combined(T,P):
    D_true = T[:,0,0,0] # Diffusion
    D_pred = P
    
    D_true_conv = D_true**2*57
    D_pred_conv = D_pred**2*57
    
    return mae(D_true_conv,D_pred_conv)#100*mae(D_true,D_pred) +


def diffusion_loss_Huber(T,P):
    D_true = T[:,0,0,0]**2*57 # Diffusion
    D_pred = P**2*57
    
    return huber(D_true,D_pred) 

def zeroloss(T,P):
    return tf.constant(0)

def mask_loss(ytrue,ypred):    
    
    T = ytrue[...,1]
    P = ypred[...,-1]

    loss1=-K.mean(T*K.log(P+1e-3)+(1-T)*K.log(1-P+1e-3))

    return loss1

def generateValDataExosomes():

        val_path='../Data/Preprocessed experimental_noise_plus_simulated_exosome/temp/intensity/'

        try:
            del valImgs
            del valLabels
        except:
           pass

        valFiles = os.listdir(val_path)
        nrOFValFiles = len(valFiles)
        #nrOFValFiles = 240
        #valFiles= [valFile for valFile in valFiles if not "iOC0.05" in valFile and not "iOC0.01" in valFile]# and not "iOC0.01" in valFile]#hacky fix to remove iOC==500, seems to produce nan values
        #valFiles = np.random.choice(valFiles,size=nrOFValFiles,replace=False)
        for i in range(0,nrOFValFiles):
    
            file = np.expand_dims(np.load(val_path+valFiles[i]),-1)          
           
            intensity = float(valFiles[i][valFiles[i].index("iOC")+3:valFiles[i].index("_")])*10000             

            if intensity >= I1 and intensity <= I2:
                try:
                    valImgs=np.append(valImgs,np.zeros((1,512,512,1)),0)
                    valLabels=np.append(valLabels,np.zeros((1,16,16,2)),0)
                except:
                    valImgs = np.zeros((1,512,512,1))
                    valLabels = np.zeros((1,16,16,2))
                
                startInd =np.random.randint(0,high=10000-512)
                valImgs[-1,...]  = np.copy(file[startInd:512+startInd,64:-64,:])
                mask = np.squeeze(skimage.measure.block_reduce(valImgs[-1,...],(32,32,1),np.mean))    
                label = np.ones((16,16))*intensity
                valLabels[-1,:,:,1] = mask
                valLabels[-1,:,:,0] = label
            
        return valImgs,valLabels
  

def GenerateValData(generateTrajectories=True):
    try:
        del valImgs
        del valLabels
        del trajLabels
    except:
        pass
    if median_process:
        val_path='../Data/Preprocessed Simulated Data Median/alliOC/intensity/'   
        traj_path = '../Data/Simulated Data - Ground Truth/'
    else:
        val_path='../Data/Preprocessed Simulated Data/alliOC/intensity/'   
        traj_path = '../Data/Simulated Data - Ground Truth/'
    valFiles = os.listdir(val_path)
    


    #valFiles = [file for file in valFiles if "D10" not in file] #Do a small-diff-pred-network
    nrOFValFiles = int(len(valFiles))#int(len(valFiles)/16)
    valFiles = np.random.choice(valFiles,size=nrOFValFiles,replace=False)
    
    
    
    
    for i in range(0,nrOFValFiles):
    
        file = np.expand_dims(np.load(val_path+valFiles[i]),-1)          
    
        intensity = float(valFiles[i][valFiles[i].index("iOC")+3:valFiles[i].index("_M")])*10000 
            
        if intensity >= I1 and intensity <= I2:
            try:
                valImgs=np.append(valImgs,np.zeros((1,8192,128,1)),0)
                valLabels=np.append(valLabels,np.zeros((1,256,4,2)),0)
            except:
                valImgs = np.zeros((1,8192,128,1))
                valLabels = np.zeros((1,256,4,2))
            
            #startInd =np.random.randint(0,high=10000-512)
            valImgs[-1,...]  = np.copy(file[904:8192+904,11:-11,:])
            mask = np.squeeze(skimage.measure.block_reduce(valImgs[-1,...],(32,32,1),np.mean))    
            label = np.ones((256,4))*intensity
            
            valLabels[-1,:,:,0] = label
    
            if generateTrajectories:
                try:
                    trajLabels=np.append(trajLabels,np.zeros((1,8192,128,1)),0)
                except:
                    trajLabels = np.zeros((1,8192,128,1))
                
                traj_file = np.load(traj_path+valFiles[i])      
                mask = np.expand_dims(skimage.measure.block_reduce(traj_file,(T_reduction_factor,L_reduction_factor),np.mean),-1)
                
                mask = mask-1
                mask = np.abs(mask/np.min(mask))
                mask = mask[904:8192+904,11:-11,:]
                trajLabels[-1,:,:,0] = mask[:,:,0]
                mask = np.squeeze(skimage.measure.block_reduce(mask,(32,32,1)))             
                valLabels[-1,:,:,1] = mask

    return valImgs, valLabels,trajLabels

def GenerateValDataWithSize(valImgs,valLabels,trajLabels,size):

    nrOFValFiles = valImgs.shape[0]
    newValImgs = np.zeros((nrOFValFiles,size,128,1))
    newValLabels = np.zeros((nrOFValFiles,int(size/32),int(128/32),2))

    for i in range(0,nrOFValFiles):
        label = trajLabels[i,...]
        flag=True
        while flag:
            startIndex = np.random.randint(0,high=valImgs.shape[1]-(size+1))
            if (label[startIndex:size+startIndex,11:-11,0] > 0).any():
                flag = False
            
    
        newValImgs[i,...]  = valImgs[i,startIndex:size+startIndex,:,:]     
       

        #  mask = np.squeeze(skimage.measure.block_reduce(newValImgs[i,...],(downsampling_factor,downsampling_factor,1),np.mean))
        mask = np.squeeze(skimage.measure.block_reduce(trajLabels[i,startIndex:size+startIndex,:,:],(32,32,1),np.mean))
            
        newValLabels[i,:,:,1] = mask
        newValLabels[i,:,:,0] = valLabels[i,0:int(size/32),:,0]

    return newValImgs, newValLabels




