import os
import numpy as np
import json
from PIL import Image, ImageDraw
from multiprocessing import Pool
import multiprocessing as mp

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    

    def kmeans(data, k):
        mn = np.mean(data, 0)
        st = np.std(data, 0)
        data = (data - mn)/st
        np.random.shuffle(data)
        centers = data[:k]

        for i in range(500):
            dist = (data[:, :, None] - np.transpose(centers)[None, :, :])**2
            cls = np.argmin(np.sum(dist, axis=1), axis=1)
            nc = []
            for j in range(k):
                nc.append(np.mean(data[cls == j, :], axis=0))
            nc = np.array(nc)
            if (nc == centers).all():
                break
            centers = nc
            
        centers = centers * st + mn
        return centers
    
    points = []
    for i in range(len(I)):
        for j in range(len(I[i])):
            if I[i,j,0] > 150 and I[i,j,1] < 0.7 * I[i,j,0] and I[i,j,2] < 0.7 * I[i,j,0]:
                points.append([i, j])
                
    points = np.array(points)
    if len(points) <= 20:
        # don't do anything if 20 px or less)
        return bounding_boxes

    # 4 boxes
    ctrs = kmeans(points, 4)
    for center in ctrs:
        bounding_boxes.append([center[1] - 10, center[0] - 10, center[1] + 10, center[0] + 10])
    
    return bounding_boxes


def process(fname):
    # read image using PIL:
    I = Image.open(fname)
    
    preds = detect_red_light(np.asarray(I))

    rdraw = ImageDraw.Draw(I)
    
    for rc in preds:
        rdraw.rectangle(rc, outline="green")
    return preds, I

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './hw01_preds/clust' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}

fnames = [os.path.join(data_path, file_names[i]) for i in range(len(file_names))]
p = Pool(mp.cpu_count())
res = p.map(process, fnames)

for i in range(len(file_names)):
    res[i][1].save(os.path.join(preds_path, file_names[i]))
    preds[file_names[i]] = res[i][0]
    
# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
