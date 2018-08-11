# This combines the images processed by preprocess.py and combine them into a singly numpy array

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

img_dir = 'resources/essex images/processed'
files = [x for x in os.listdir(img_dir) if x[-4:] == '.png']
dataset = np.empty((len(files),64,64,1))
for i,file in tqdm(enumerate(files)):
      img = Image.open(os.path.join(img_dir, file))
      img = np.array(img)
      if len(img.shape) > 2:
            img = img[:,:,0]
      dataset[i] = np.expand_dims(img,axis=2)
np.save('dataset.npy', dataset)