# This recursively searches a folder for images, resizes them, converts them if necessary, and saves them to different folder

import os
from os.path import isdir
from PIL import Image

def convert_to_png(folder, dest_folder, from_ending=None, to_ending=None):
      if not isdir(folder):
            if folder[-4:] == from_ending or from_ending is None:
                  img = Image.open(folder)
                  img = img.resize((28,28))
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
                  convert_to_png(os.path.join(folder,file),
                                 dest_folder,
                                 from_ending=from_ending,
                                 to_ending=to_ending)

convert_to_png('resources\\raw\\CroppedYale', 'resources\\processed\\all', '.pgm', '.png')