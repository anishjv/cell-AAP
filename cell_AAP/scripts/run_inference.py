import cell_AAP.scripts.inference as inf # type:ignore
from pathlib import Path


model_name = 'HeLa' # can be one of ['HeLa', 'U2OS']
confluency_est = 1800 # can be in the interval (0, 2000]
conf_threshold = .275 # can be in the interval (0, 1)
movie_file = Path('/Users/whoisv/Library/CloudStorage/GoogleDrive-anishjv@umich.edu/.shortcut-targets-by-id/1Um2WOlVPLN717lyJFpg0Q21oifwMs_7n/IXN images/ajit_talk/20221026_A4_s1_phs.tif')
interval = [0, 1]
save_dir = Path('/Users/whoisv/Desktop/')

def main():
    container = inf.configure(model_name, confluency_est, conf_threshold, save_dir)
    result = inf.run_inference(container, movie_file, interval)
    inf.save(container, result)

if __name__ == "__main__":
    main()
