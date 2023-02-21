from deeptrack.features import Feature
import numpy as np
from scipy.signal import convolve2d



class init_particle_counter(Feature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, image, **kwargs):
        # Init particle counter
        nbr_particles = 0
        image.append({"nbr_particles":nbr_particles})
        
        return image
    def __init__(self, vel=0, D=0, I=0, s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        
        # Init particle counter
        nbr_particles = 0
        image.append({"nbr_particles":nbr_particles})
        
        return image

class Trajectory(Feature):
    def __init__(self, vel=0, diffusion=[0.1, 2], intensity=[0.01, 5], std=[0.04,0.05], **kwargs):
        """
        This feature generates a single-particle trajectory with intensity fluctuations.
        
        Args:
            vel (float): velocity of the particle.
            diffusion (list of float): Range of diffusion coefficients of the particle.
            intensity (list of float): Range of intensities of the particle.
            s (float): Standard deviation of gaussian intensity fluctuations.
        """
        super().__init__(**kwargs)
        self.intensity = intensity
        self.diffusion = diffusion
        self.vel = vel
        self.std = std

    def get(self, image, **kwargs):
        I = self.intensity[0] + self.intensity[1] * np.random.rand()
        D = self.diffusion[0] + (self.diffusion[1] - self.diffusion[0]) * np.random.rand()
        s = self.std[0] + (self.std[1] - self.std[0]) * np.random.rand()
        # Convert I and D to appropriate scales
        I /= 10000
        D /= 10

        # Particle counter
        nbr_particles = image.properties[1]['nbr_particles']
        particle_index = nbr_particles + 1
        image.properties[1]['nbr_particles'] += 1

        # Generate particle trajectory
        length = image.shape[1]
        times = image.shape[0]
        x = np.linspace(-1, 1, length)
        t = np.linspace(-1, 1, times)
        X, Y = np.meshgrid(t, x)

        # Gaussian function to model the particle trajectory
        f2 = lambda a, x0, s, b, x: a * np.exp(-(x - x0)**2/s**2) + b

        # Add some randomness to the initial particle position
        x0 = -1 + 2 * np.random.rand()

        # Update particle positions using the Gaussian function and random diffusion
        x0 += np.cumsum(self.vel + D * np.random.randn(times))
        v1 = np.transpose(I * f2(1, x0, s, 0, Y))

        # Subtract the particle intensity from the image intensity to simulate its response
        image[..., 0] *= (1 - v1)

        # Store the particle trajectory
        particle_trajectory = np.transpose(f2(1, x0, s, 0, Y))

        # Add the trajectory to the image
        image[..., 1] += particle_trajectory

        # Clip trajectories so that overlapping trajectories don't take on values > 1
        image[..., 1] = np.clip(image[..., 1], 0, 1)

        # Save single trajectory as additional image
        image[...,-particle_index] = particle_trajectory      

        # Save particle information to image properties
        pixel_size = 0.0295
        try:
            image.properties["D"] += 10 * D
            image.properties["I"] += s * I * np.sqrt(2 * np.pi) * 256 * pixel_size
        except:
            image.append({"D": 10 * D, "I": s * I * np.sqrt(2 * np.pi) * 256 * pixel_size})

        return image
   
  
class GenNoise(Feature):
    def __init__(self, noise_lev=0, dX=0, dA=0,biglam=0,bgnoiseCval=0,bgnoise=0,bigx0=0,sinus_noise_amplitude=0,freq=0,**kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA,biglam=biglam,bgnoiseCval=bgnoiseCval,bgnoise=bgnoise,bigx0=bigx0,sinus_noise_amplitude=sinus_noise_amplitude,freq=0, **kwargs
        )

    def get(self, image, noise_lev, dX, dA,biglam,bgnoiseCval,bgnoise,bigx0,sinus_noise_amplitude,freq, **kwargs):
        from scipy.signal import convolve
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        bgnoise*=np.random.randn(length)
        
        ll=(np.pi-.05)
        # Compute the sinusoidal noise
        j = np.arange(times)
        dx = (sinus_noise_amplitude * np.random.randn(times) + np.sin(freq*j)) * dX

        # Compute the background noise correlation function
        x = np.linspace(-1, 1, length)
        bgnoiseC = np.array([f2(1, 0, bgnoiseCval, dx[t], x) for t in range(times)])

        # Normalize the background noise correlation function
        bgnoiseC /= np.sum(bgnoiseC, axis=1, keepdims=True)

        # Compute the background noise
        bg = [f2(1, bigx0 + dx[t], biglam, 0, x) * (1 + convolve(bgnoise, bgnoiseC[t], mode="same")) for t in range(times)]

        # Compute the amplitude noise and add it to the background
        dAmp0 = dA * np.random.randn(times)
        bg *= (1 + dAmp0[:, np.newaxis])

        # Add the noise to the image
        noise = noise_lev * np.random.randn(times, length) + 0.4 * noise_lev * np.random.randn(times, length)
        image[:,:,0] = bg * (1 + noise)
        
        return image
    
class PostProcess(Feature):
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(noise_lev=noise_lev, dX=dX, dA=dA, **kwargs)

    def get(self, image, **kwargs):

        length = image.shape[1]
        times = image.shape[0]
        x = np.linspace(-1, 1, length)
        t = np.linspace(-1, 1, times)
        X, Y = np.meshgrid(t, x)

        # Subtract mean and normalize intensity of the first channel (background noise)
        # along each column
        image[:,:,0] = (image[:,:,0] - np.expand_dims(np.mean(image[:,:,0], axis=0), axis=0)) / np.mean(image[...,0], axis=0)

        # Compute a 200x1 box filter (convolution kernel) and subtract from the first channel
        ono = np.ones((200, 1))
        ono = ono / np.sum(ono)

        image[:,:,0] -= convolve2d(image[:,:,0], ono, mode="same")
        image[:,:,0] -= convolve2d(image[:,:,0], np.transpose(ono), mode="same")


        # Subtract mean again after filtering
        image[:,:,0] -= np.expand_dims(np.mean(image[:,:,0], axis=0), axis=0)

        # Normalize intensity of the first channel along each column by standard deviation
        a = np.std(image[...,0], axis=0)
        image[:,:,0] /= a
        
        try:
            image.properties["I"] /= a
        except:
            pass
        
        # Return the processed image
        return image  

class post_process_basic(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA, **kwargs
        )

    def get(self, image, **kwargs):              
        #Perform same preprocessing as done on experimental images
        image[:,:,0]=(image[...,0]-np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0))/np.mean(image[...,0],axis=0)
        return image

    
class input_array(Feature):
    __distributed__ = False
    def __init__(self, times=512, length=512, **kwargs):
        super().__init__(
            times=times, length=length, **kwargs
        )
    def get(self,image, times, length, **kwargs):
        image=np.zeros((times,length,10))
        return image
    
    
def heaviside(a):
    a[a>0] = 1
    a[a!=1] = 0
    return a

class get_diffusion(Feature):
    
    def __init__(self, vel=0, D=0, I=0, s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        LOW = 0.1
        HIGH = 1.15
        D = (LOW + HIGH*np.random.rand())
        image.append({"D":D})
        
        return image
    
class get_long_DNA_trajectory(Feature):

    
    def __init__(self, vel=0, D=0.1, I=0.1,I_2=0.5,particle_width=0.1,s=0.01, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,I_2=I_2,particle_width=particle_width,s=s, nbr_particles=0, **kwargs
        )

    def get(self, image, vel, D, I,I_2, particle_width, s, **kwargs):
        
        # vel= np.random.choice([(500+500*np.random.rand())*10**-6,(1500+1000*np.random.rand())*10**-6])
        # D=  0.0005+0.0005*np.random.rand()
        # I=5
        # I_2=20
        # particle_width =  0.08 + 0.04*np.random.rand()
        
        from scipy.signal import convolve
        I = I/10000
        # Particle counter
        nbr_particles = image.properties[1]['nbr_particles']
        particle_index = nbr_particles + 1
        image.properties[1]['nbr_particles'] += 1
        
        f=lambda a,b,x0,x: a*(heaviside((x-x0)-b/2)-heaviside((x-x0)+b/2))
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        x0=-1+2*np.random.rand()
        x0+=np.cumsum(vel+D*np.random.randn(times))

        x1= x0-particle_width/2+np.random.rand()*particle_width
        
        vv=np.transpose(convolve(I*f(1,particle_width,x0,Y),f2(1,0,s,0,Y[:,0:1]),mode="same"))
        v1=np.array(vv)-np.transpose(I*I_2*f2(1,x1,s,0,Y))         

        
        image[...,0]*=(1+v1)
        image[...,2]*=(1+v1)
        
        particle_trajectory =np.transpose(convolve(f(1,particle_width,x0,Y),f2(1,0,s,0,Y[:,0:1]),mode="same"))#v1

        # Add trajectory to image
        image[...,1] += particle_trajectory 
        image[...,1]=np.clip(np.abs(image[...,1]),0,1)
        # Save single trajectory as additional image
        image[...,-particle_index] = particle_trajectory             
        
        try:
            image.properties["D"]+=10*D#*np.sum(np.transpose(f2(1,x0,.1,0,Y)))
            image.properties["I"]+=s*I*np.sqrt(2*np.pi)*256*.03
        except:
            image.append({"D":10*D,"hash_key":list(np.zeros(4))})
            image.append({"I":s*I*np.sqrt(2*np.pi)*256*.03,"hash_key":list(np.zeros(4))})
            
        return image


