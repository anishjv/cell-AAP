import skimage
import cell_AAP.annotation.annotation_utils as au

"Standard Configurations for HeLa cell data"

_HELA = {}
_HELA['VERSION'] = 1
_HELA['THRESHOLD_TYPE'] = 'single'
_HELA['THRESHOLD_DIVISION'] = 0.75
_HELA['TOPHATSTRUCT'] = skimage.morphology.square(71)
_HELA['EROSIONSTRUCT'] = skimage.morphology.disk(8)
_HELA['GAUSSIAN_SIGMA'] = 2
_HELA['POINTPROMPTS'] = True
_HELA['BOXPROMPTS'] = False
_HELA['PROPSLIST'] = [
    'area', 'solidity', 'perimeter_crofton', 'area_convex', 
    'eccentricity', 'axis_major_length', 'axis_minor_length',
    'perimeter', 'centroid_local', 'euler_number',
    'extent', 'intensity_max', 'intensity_min', 'area_filled'
    ]
_HELA['FRAMESTEP'] = 1
_HELA['BOX_SIZE'] = (au.get_box_size, (2.5,))
_DEFAULT = _HELA


"Standard Configurations for U20S cell data"
#TODO
#Add standard configurations 

"Standard Configurations for HT1080 cell data"
#TODO
#Add standard configurations 

"Standard Configurations for Yeast cell data"
#TODO
#Add standard configurations 