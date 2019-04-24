'''
Within this script, focus to pass set of frames to the siamese model.
----
If need to run this script seperately, then can edit the relevant input file path and output file path.

If need to use this script within another code then can import the script and call the functions with relevant arguments.

We define the positive and negative as follows for this project:
    pos + pos -> 1
    pos + neg -> 0
'''


import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

def import_frames_to_siamese(input_frames_path, object_of_interest_path, image_size):

    if (os.path.isdir(input_frames_path) and os.path.isdir(object_of_interest_path)):

        video_frames_array = []
        video_frames_path_array = []
        object_of_interest_frame_array = []

        frame_list = [file for file in os.listdir(input_frames_path) if isfile(join(input_frames_path, file))]
        frame_list.sort(key=lambda x: int(x[5:-4]))
        print("frame_list")
        print(frame_list)
        object_of_interest = [file1 for file1 in os.listdir(object_of_interest_path) if isfile(join(object_of_interest_path, file1))]

        if(len(frame_list)!=0):
            print("Loading the frames list.")
            for i in range(len(frame_list)):

                img_frame = load_image(input_frames_path, frame_list[i], image_size)
                video_frames_path_array.append(plt.imread(os.path.join(input_frames_path, frame_list[i])).astype(np.float32))
                video_frames_array += [img_frame]

                ooi_frame = load_image(object_of_interest_path, object_of_interest[0], image_size)
                object_of_interest_frame_array += [ooi_frame]

        else:
            print("Need to include frames in the required format within the given location of frames.")
    else:
        print("Given path of the frames not exists.")
    return (np.array(video_frames_array), np.array(object_of_interest_frame_array), frame_list)

def load_image(image_dir, image_name, image_size):
    image_cache = {}
    #print("IMAGE_SIZE: ")
   # print(image_size)
    if not (image_name in image_cache.keys()):
        image = plt.imread(os.path.join(image_dir, image_name)).astype(np.float32)
        #image = imresize(image, (image_size, image_size, 3))
        image = imresize(image, (image_size, image_size, 3))
        image = np.divide(image, 256)
        image_cache[image_name] = image
    return image_cache[image_name]

def run():
    input_frames_path = "./data/generated_frames/"
    object_of_interest_path = "./data/object_of_interest_path/"
    size = 224
    import_frames_to_siamese(input_frames_path, object_of_interest_path, size)

if __name__ == "__main__":
    run()
