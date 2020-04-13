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
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    def get_red(size, img):
        boxes = []
        for i in range(len(img) - 3*size):
            for j in range(len(img[i]) - size):
                #red lights are at top, so check if top 1/3 is red and bottom 1/3 is dark
                square_top = img[i:i + size, j:j + size]
                rect_bottom = img[i + size:i + 3*size, j:j + size]
                avg = np.mean(np.mean(square_top, 0), 0)
                avg_r = np.mean(np.mean(rect_bottom, 0), 0)
                if avg[0] > 150 and avg[1] < 0.7 * avg[0] and avg[2] < 0.7 * avg[0] and np.mean(avg_r) < 50:
                    boxes.append([j, i, j + size, i + 3 * size])
                
        return boxes

    return get_red(10, I)

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
preds_path = './hw01_preds/simple' 
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
