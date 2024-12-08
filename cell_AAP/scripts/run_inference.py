import cell_AAP.scripts.inference as inf # type:ignore
from pathlib import Path


model_name = 'HeLa' # can be on of ['Hela', 'U2OS']
confluency_est = 1800 # can be in the interval (0, 2000]
conf_threshold = .275 # can be in the interval (0, 1)
movie_file = Path('/Users/whoisv/Library/CloudStorage/GoogleDrive-anishjv@umich.edu/.shortcut-targets-by-id/1Um2WOlVPLN717lyJFpg0Q21oifwMs_7n/IXN images/ajit_talk/20221026_A4_s1_phs.tif')
interval = [0, 1]

def main():
    container = inf.configure(model_name, confluency_est, conf_threshold)
    result = inf.run_inference(container, movie_file, interval)

    print(result['name'])
    print(result['semantic_movie'].shape)
    print(result['instance_movie'].shape)
    print(result['centroids'].shape)
    print(len(result['confidence']))

    return result

if __name__ == "__main__":
    main()
