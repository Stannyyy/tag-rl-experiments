import glob
import os
import cv2
import numpy as np

def record_game(game_name, path = r'C:\Users\Stanny\OneDrive - Trifork B.V\Documents\Tag'):
    # Fetch all images
    all_images = [im_path for im_path in glob.glob(os.path.join(path,'*.png'))]
    img_array = []

    # Loop over images and attach
    for filename in all_images:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # Output to video
    out = cv2.VideoWriter(os.path.join(path,game_name+'.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

    # Now delete all images
    [os.remove(im_path) for im_path in all_images]