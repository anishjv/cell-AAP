import os 
import cv2
from PIL import Image
import numpy as np
#TODO
#cannot import this module without installing cell-AAP, this should not be the case, throws "no module 'cell-AAP' error"
from cell_AAP.annotation import annotation_utils #type:ignore



def write_coco_conv_dataset(parent_dir : str, phase_image_stack, segmentations, labeled_data_frame, name : str, label_to_class : dict, bin_size:tuple = (1024, 1024), bin_method :str = 'max'):
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
    
    train_cutoff = labeled_data_frame[-int(labeled_data_frame.shape[0] // (10/7)), -3]
    val_cutoff = labeled_data_frame[int(labeled_data_frame.shape[0] // 10), -3]

    main_path = os.path.join(parent_dir, f'{name}')
    os.mkdir(main_path)       
    os.chdir(main_path)
    train_path = os.path.join(main_path, 'train')
    test_path = os.path.join(main_path, 'test')
    val_path = os.path.join(main_path, 'val')

    for i in [train_path, val_path, test_path]:
        os.mkdir(i)
        os.chdir(i)
        image_path = os.path.join(i, 'images')
        os.mkdir(image_path)
        annotation_path = os.path.join(i, 'annotations')
        os.mkdir(annotation_path)
    

    for j in range(labeled_data_frame.shape[0]):
        if labeled_data_frame[j, -3] >= train_cutoff:
            os.chdir(os.path.join(train_path, 'annotations'))
        elif labeled_data_frame[j, -3] <= val_cutoff:
            os.chdir(os.path.join(val_path, 'annotations'))
        else:
            os.chdir(os.path.join(test_path, 'annotations'))

        mask = np.unpackbits(
                segmentations[int(labeled_data_frame[j, -3])][int(labeled_data_frame[j, -2])],
                axis = 0,
                count = 2048
        )
        mask = mask * 255
        mask  = annotation_utils.binImage(mask, bin_size, bin_method)
        if labeled_data_frame[j, -1] == 0:
            cv2.imwrite(
                        f'{int(labeled_data_frame[j, -3])}_{label_to_class[0]}_frame{int(labeled_data_frame[j, -3])}cell{int(labeled_data_frame[j, -2])}.png', 
                        mask
                        )
        elif labeled_data_frame[j, -1] in [1, 2]:
            cv2.imwrite(
                        f'{int(labeled_data_frame[j, -3])}_{label_to_class[1]}_frame{int(labeled_data_frame[j, -3])}cell{int(labeled_data_frame[j, -2])}.png', 
                        mask
                        )  

    

    for k in range(int(max(labeled_data_frame[:, -3]))):
        if k >= train_cutoff:
            os.chdir(os.path.join(train_path, 'images'))
        elif k <= val_cutoff:
            os.chdir(os.path.join(val_path, 'images'))
        else:
            os.chdir(os.path.join(test_path, 'images'))

        image = Image.fromarray( annotation_utils.bw_to_rgb(phase_image_stack[k]) )
        image = annotation_utils.binImage(image, bin_size, bin_method)
        image.save(f'{k}.jpg')
        
    
    
                   
                   
    
    
    