import sys, tifffile
from pathlib import Path
import inference as inf
from skimage.io import imread
import numpy as np

import sys, tifffile
from pathlib import Path
import inference as inf
from skimage.io import imread
import numpy as np

model_name = 'HeLa_focal'
confluency_est = 1800 # can be in the interval (0, 2000]
conf_threshold = .25 # can be in the interval (0, 1)

# folder definition
root_folder = Path('/nfs/turbo/umms-ajitj/anishjv/cyclinb_analysis/20251028-cycb-gsk')
save_dir = Path('/nfs/turbo/umms-ajitj/anishjv/cyclinb_analysis/20251028-cycb-gsk')
filter_str  = '*_phs.tif'

blanks = ['G', 'H']

file_list = []
for phs_file_name in root_folder.glob(filter_str):
    phs_file_name = str(phs_file_name)
    if any(stub in phs_file_name for stub in blanks):
        pass
    else:
        file_list.append(phs_file_name)
        
num_files = len(file_list)
print(f'Found {num_files} files')

# Following code is modified from Ajit P. Joglekar

def main():

    container = inf.configure(model_name, confluency_est, conf_threshold, save_dir = save_dir)
    for i in np.arange(num_files):
        phs_file = tifffile.TiffFile(file_list[i])
        interval = [0, len(phs_file.pages)-1]
        result = inf.run_inference(container, file_list[i], interval)
        inf.save(container, result)
        print(f"{file_list[i]} written!")
        print(f"{i} out of {num_files} processed")

    return


if __name__ == "__main__":
    main()