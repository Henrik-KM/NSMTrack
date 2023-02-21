import numpy as np

#If the segmentations are near-perfect, we can use our described treshold functions, negating the need for complex PyTorch YOLO integration..
def manYOLO(im,treshold=0.05,trajTreshold=256):
    nump = 1
    batchSize = 1
    YOLOLabels =np.reshape([None]*batchSize*1*5,(batchSize,1,5))
    times = im.shape[1]
    length = im.shape[2]
    for j in range(0,batchSize):
        for k in range(0,nump):
            particle_img = im[j,:,:,k]
            #particle_img = particle_img/np.max(particle_img)

            particleOccurence = np.where(particle_img>treshold)
            if np.sum(particleOccurence) <= 0:
                continue
            else:
                trajectoryOccurence = np.diff(particleOccurence[0])
                trajectories = particleOccurence[0][np.where(trajectoryOccurence>trajTreshold)]
                trajectories = np.append(0,trajectories)
                trajectories = np.append(trajectories,times)

                for traj in range(0,len(trajectories)-1): 


                    particleOccurence = np.where(particle_img[trajectories[traj]:trajectories[traj+1],:]>treshold)
                    if np.sum(particleOccurence[1]) <=0 or np.sum(particleOccurence[0]) <=0:
                        continue
                    constant = trajectories[traj]
                    if traj != 0:
                        particleOccurence = np.where(particle_img[trajectories[traj]+trajTreshold:trajectories[traj+1],:]>treshold)
                        constant = trajectories[traj]+trajTreshold
                        
                    

                    x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                    y1,y2 = np.min(particleOccurence[0])+constant,np.max(particleOccurence[0])+constant


                    if YOLOLabels[0,0,0] is None:
                        YOLOLabels = np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5))   
                    else:
                        YOLOLabels =np.append(YOLOLabels,np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5)),1)
 
        return YOLOLabels


def manYOLOSplit(im,treshold=0.05,trajTreshold=16,splitTreshold=512):
    nump = 1
    batchSize = 1
    YOLOLabels =np.reshape([None]*batchSize*1*5,(batchSize,1,5))
    times = im.shape[1]
    length = im.shape[2]
    for j in range(0,batchSize):
        for k in range(0,nump):
            particle_img = im[j,:,:,k]
            #particle_img = particle_img/np.max(particle_img)

            particleOccurence = np.where(particle_img>treshold)
            if np.sum(particleOccurence) <= 0:
                continue
            else:
                trajectoryOccurence = np.diff(particleOccurence[0])
                trajectories = particleOccurence[0][np.where(trajectoryOccurence>trajTreshold)]
                trajectories = np.append(0,trajectories)
                trajectories = np.append(trajectories,times)

                traj = 0
                while traj < len(trajectories)-1:


                    particleOccurence = np.where(particle_img[trajectories[traj]:trajectories[traj+1],:]>treshold)

                    constant = trajectories[traj]
                    if traj != 0 or True:
                        particleOccurence = np.where(particle_img[trajectories[traj]+trajTreshold:trajectories[traj+1],:]>treshold)
                        constant = trajectories[traj]+trajTreshold

                    if np.sum(particleOccurence[1]) <=0 or np.sum(particleOccurence[0]) <=0:
                        traj+=1
                        continue                    

                    splitIndices = np.where(np.abs(np.diff(particleOccurence[1]))>splitTreshold)[0]
                    if len(splitIndices)>0:# and traj != 0:             
                        trajectories = np.insert(trajectories, traj+1,particleOccurence[0][splitIndices[0]]+constant)
                        continue
                    else:
                        traj+=1


                    x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                    y1,y2 = np.min(particleOccurence[0])+constant,np.max(particleOccurence[0])+constant



                    if YOLOLabels[0,0,0] is None:
                        YOLOLabels = np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5))   
                    else:
                        YOLOLabels =np.append(YOLOLabels,np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5)),1)
                        
    return YOLOLabels


def ConvertYOLOLabelsToCoord(YOLOLabels,xdim,ydim):
        #image,x1,y1,x2,y2
        coordinates = np.zeros(YOLOLabels.shape)
        coordinates[:,0] = YOLOLabels[:,0]
        coordinates[:,1] = (xdim-1)*YOLOLabels[:,1]-(xdim-1)/2*YOLOLabels[:,3]
        coordinates[:,2] = (ydim-1)*YOLOLabels[:,2]-(ydim-1)/2*YOLOLabels[:,4]
        coordinates[:,3] = (xdim-1)*YOLOLabels[:,1]+(xdim-1)/2*YOLOLabels[:,3]
        coordinates[:,4] = (ydim-1)*YOLOLabels[:,2]+(ydim-1)/2*YOLOLabels[:,4]

        return coordinates

def ConvertCoordToYOLOLabels(coordinates,xdim,ydim):
    #image,xc,yc,xw,yw
    labels = np.zeros(coordinates.shape)
    labels[:,0] = coordinates[:,0]
    labels[:,1] = (coordinates[:,1]+coordinates[:,3])/2/(xdim-1)
    labels[:,2] = (coordinates[:,2]+coordinates[:,4])/2/(ydim-1)
    labels[:,3] = (coordinates[:,3]-coordinates[:,1])/(xdim-1)
    labels[:,4] = (coordinates[:,4]-coordinates[:,2])/(ydim-1)
    
    return labels