# This recursively searches a folder for images, resizes them, converts them if necessary, and saves them to different folder

import os
from PIL import Image

def process(folder, dest_folder, target_shape=None, from_ending=None, to_ending=None,grayscale=False):
      if dest_folder.startswith(folder):
            raise ValueError('dest_folder cannot be a subfolder of folder!')
      if os.path.isfile(folder):
            try:
                  if from_ending is None or folder.endswith(from_ending):
                        img = Image.open(folder)
                        if target_shape is not None:
                              img = img.resize(target_shape)
                        if grayscale:
                              img = img.convert('L')
                        if to_ending is not None:
                              ending = to_ending
                        else:
                              ending = folder[folder.rfind('.'):]
                        num_processed = len(os.listdir(dest_folder))
                        img_name = str(num_processed) + ending
                        img_name = os.path.join(dest_folder, img_name)
                        img.save(img_name)
            except Exception as e:
                  print('Failed for file %s: %s' % (folder, str(e)))
      elif os.path.isdir(folder):
            files = os.listdir(folder)
            for file in files:
                  process(os.path.join(folder,file),
                          dest_folder,
                          target_shape=target_shape,
                          from_ending=from_ending,
                          to_ending=to_ending,
                          grayscale=grayscale)
      else:
            print('ERROR: %s does not exist!' % folder)

process(r'resources\essex images\raw',
        r'resources\essex images\processed',
        target_shape=(64,64),
        from_ending='.jpg',
        to_ending='.png',
        grayscale=True)