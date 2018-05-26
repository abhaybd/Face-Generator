# This recursively searches a folder for images, resizes them, converts them if necessary, and saves them to different folder

import os
from PIL import Image

def process(folder, dest_folder, target_shape=None, from_ending=None, to_ending=None):
      if dest_folder.startswith(folder):
            raise ValueError('dest_folder cannot be a subfolder of folder!')
      if not os.path.isdir(folder):
            if from_ending is None or folder[folder.rfind('.'):] == from_ending:
                  img = Image.open(folder)
                  if target_shape is not None:
                        img = img.resize(target_shape)
                  img_name = folder
                  if to_ending is not None:
                        img_name = img_name[img_name.rfind(os.sep)+1:img_name.rfind('.')] + to_ending
                  else:
                        img_name = img_name[img_name.rfind(os.sep)+1:]
                  img_name = os.path.join(dest_folder, img_name)
                  img.save(img_name)
                  os.remove(folder)
      else:
            files = os.listdir(folder)
            for file in files:
                  process(os.path.join(folder,file),
                          dest_folder,
                          target_shape=target_shape,
                          from_ending=from_ending,
                          to_ending=to_ending)

process('resources\\raw\\CroppedYale',
        'resources\\processed\\all',
        target_shape=(28,28),
        from_ending='.pgm',
        to_ending='.png')