import os 
import cv2
from PIL import Image
import numpy as np
from data_module.data_utils import *



def write_coco_conv_dataset(parent_dir, phase_image_stack, segmentations, labeled_data_frame, name, label_to_class : dict):
    '''
    Saves annotations(masks) and images in a manner that can be converted to COCO format using common tools
    -------------------------------------------------------------------------------------------------------
    INPUTS:
            parent_dir: string, directory which folders are to be created within
            phase_image_stack: n-darray, array containing phase images from which annotions come from
            segmentations: n-darray, rank 4 tensor indexed as segmentations[mu][nu] where mu references a frame and nu a cell:
                          contains masks with each mask corresponding to an annotation. (Must be unpacked from bitmap repr)
            labeled_data_frame: n-darray, dataframe containing region props and classifications for each cell
            name: string, name of dataset to be created
            label_to_class: dict, dictionary containing int to string key value pairs, specifying what number classification corresponds to 
                                  what verbal classification, i.e 0-> mitotic

    '''
    
    main_path = os.path.join(parent_dir, f'{name}')
    os.mkdir(main_path)       
    os.chdir(main_path)
    image_path = os.path.join(main_path, 'images')
    os.mkdir(image_path)
    annotation_path = os.path.join(main_path, 'annotations')
    os.mkdir(annotation_path)
    
   
    os.chdir(annotation_path)
    for j in range(labeled_data_frame.shape[0]):
        mask = np.unpackbits(
                segmentations[int(labeled_data_frame[j, -3])][int(labeled_data_frame[j, -2])],
                axis = 0,
                count = 2048
        )
        mask = mask * 255
        if labeled_data_frame[j, -1] == 0:
            cv2.imwrite(
                        f'{int(labeled_data_frame[j, -3])}_{label_to_class[0]}_frame{int(labeled_data_frame[j, -3])}cell{int(labeled_data_frame[j, -2])}.png', 
                        mask * 255
                        )
        elif labeled_data_frame[j, -1] in [1, 2]:
            cv2.imwrite(
                        f'{int(labeled_data_frame[j, -3])}_{label_to_class[1]}_frame{int(labeled_data_frame[j, -3])}cell{int(labeled_data_frame[j, -2])}.png', 
                        mask
                        )  
    
        
    
    os.chdir(image_path)
    frames = list(range(0, labeled_data_frame.shape[0]))
    for k in frames:
        image = Image.fromarray( dt.bw_to_rgb(phase_image_stack[k]) )
        image.save(f'{k}.jpg')
        
    
    
                   
                   
    
    
    