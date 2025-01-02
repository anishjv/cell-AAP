import sys, tifffile
from pathlib import Path
sys.path.append('/Users/ajitj/Google Drive/ImageAnalysis/cell_analysis')
import inference as inf
from skimage.io import imread
import numpy as np

model_name = 'HeLa' # can be on of ['HeLa', 'U2OS']
confluency_est = 1800 # can be in the interval (0, 2000]
conf_threshold = .275 # can be in the interval (0, 1)

# folder definition
root_folder = Path('/Users/ajitj/Desktop/current/test')
save_dir = Path('/Users/whoisv/Desktop/')
filter_str  = '*_.tif'


# Following code is modified from Ajit P. Joglekar
file_list = []
for phs_file_name in root_folder.glob(filter_str):
    file_list.append(phs_file_name)
num_files = len(file_list)

def main():
    container = inf.configure(model_name, confluency_est, conf_threshold)
    for i in np.arange(num_files):
        phs_file = tifffile.TiffFile(file_list[i])
        interval = [0, len(phs_file.pages)-1]
        result = inf.run_inference(container, phs_file_name, interval)
        inf.save(container, result)
        print(f"{phs_file_name} written!")
        print(f"{i} out of {num_files} processed")

    return


if __name__ == "__main__":
    main()