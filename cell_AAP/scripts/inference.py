import pooch
import numpy as np
import re
from detectron2.engine import DefaultPredictor
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from typing import Optional
import torch
import cell_AAP.annotation.annotation_utils as au  # type:ignore
from skimage.morphology import binary_erosion, disk
import skimage.measure
import tifffile as tiff
import os
import pandas as pd



def color_masks(
    segmentations: np.ndarray,
    labels,
    method: Optional[str] = "random",
    custom_dict: Optional[dict[int, int]] = None,
    erode = False
) -> np.ndarray:
    """
    Takes an array of segmentation masks and colors them by some pre-defined metric. If metric is not given masks are colored randomely
    -------------------------------------------------------------------------------------------------------------------------------------
    INPUTS:
        segmentations: np.ndarray
        labels: np.ndarray
        method: str
        custom_dict: dict
    OUTPUTS:
        seg_labeled: np.ndarray
    """
    
    if method == "custom":
        try:
            assert custom_dict != None
            assert np.isin(labels, list(custom_dict.keys())).all() == True
        except AssertionError:
            print('Input labels and mapping dictionary did not match when coloring movie, reverting to straight coloring')
            method = "straight"

    if segmentations.size(dim=0) == 0:
        seg_labeled = np.zeros(
            (segmentations.size(dim=1), segmentations.size(dim=2)), dtype="uint8"
        )
        return seg_labeled

    seg_labeled = np.zeros_like(segmentations[0], int)
    for i, mask in enumerate(segmentations):
        loc_mask = seg_labeled[mask]
        mask_nonzero = list(filter(lambda x: x != 0, loc_mask))
        if len(mask_nonzero) < (loc_mask.shape[0] / 4):  # Roughly IOU < 0.5

            if method == "custom":
                seg_labeled[mask] += custom_dict[labels[i]]

            elif method == "straight":
                seg_labeled[mask] += labels[i]

            else:
                if erode == True:
                    mask = binary_erosion(mask, disk(3))
                elif labels[i] == 0:
                    seg_labeled[mask] = 2 * i
                else:
                    seg_labeled[mask] = 2 * i + 1

    return seg_labeled



def get_model(model_name:str):
    """
    Instaniates POOCH instance containing model files from the model_registry
    --------------------------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()I
    """
    url_registry = {
        "HeLa": "doi:10.5281/zenodo.15587924",
        "HeLa_focal": "doi:10.5281/zenodo.15587884",
        "HT1080_focal": "doi:10.5281/zenodo.15632609",
        "HT1080": "doi:10.5281/zenodo.15632636",
        "RPE1_focal": "doi:10.5281/zenodo.15632647",
        "RPE1": "doi:10.5281/zenodo.15632661",
        "U2OS_focal": "doi:10.5281/zenodo.15632668",
        "U2OS": "doi:10.5281/zenodo.15632681",
        "general": "doi:10.5281/zenodo.15707118",
    }

    weights_registry = {
        "HeLa": (
            "model_0040499.pth",
            "md5:62a043db76171f947bfa45c31d7984fe"
        ),
        "HeLa_focal": (
            "model_0053999.pth",
            "md5:40eb9490f3b66894abef739c151c5bfe"
        ),
        "HT1080_focal": (
            "model_0052199.pth",
            "md5:f454095e8891a694905bd2b12a741274"
        ),
        "HT1080": (
            "model_0034799.pth",
            "md5:e5ec71a532d5ad845eb6af37fc785e82"
        ),
        "RPE1_focal": (
            "model_final.pth",
            "md5:f3cc3196470493bba24b05f49773c00e"
        ),
        "RPE1": (
            "model_0048299.pth",
            "md5:5d04462ed4d680b85fd5525d8efc0fc9"
        ),
        "U2OS_focal": (
            "model_final.pth",
            "md5:8fbe8dab57cd96e72537449eb490fa6f"
        ),
        "U2OS": (
            "model_final.pth",
            "md5:8fbe8dab57cd96e72537449eb490fa6f"
        ),
        "general": (
            "model_0061499.pth",
            "md5:62e5f4be12227146f6a9841ada46526a"
        )

    }

    configs_registry = {
        "HeLa": (
            "config.yaml",
            "md5:320852546ed1390ed2e8fa91008e8bf7",
            "lazy"
        ),
        "HeLa_focal": (
            "config.yaml",
            "md5:320852546ed1390ed2e8fa91008e8bf7",
            "lazy"
        ),
        "HT1080_focal": (
            "config.yaml",
            "md5:cea383632378470aa96dc46adac5d645",
            "lazy"
        ),
        "HT1080": (
            "config.yaml",
            "md5:71674a29e9d5daf3cc23648539c2d0c6",
            "lazy"
        ),
        "RPE1_focal": (
            "config.yaml",
            "md5:78878450ef4805c53b433ff028416510",
            "lazy"
        ),
        "RPE1": (
            "config.yaml",
            "md5:9abb7fcafdb953fff72db7642824202a",
            "lazy"
        ),
        "U2OS_focal": (
            "config.yaml",
            "md5:ab202fd7e0494fce123783bf564a8cde",
            "lazy"
        ),
        "U2OS": (
            "config.yaml",
            "md5:2ab6cd0635b02ad24bcb03371839b807",
            "lazy"
        ),
        "general": (
            "config.yaml",
            "md5:ad609c147ea2cd7d7fde0d734de2e166",
            "lazy"
        )

    }

    model = pooch.create(
        path=pooch.os_cache("cell_aap"),
        base_url=url_registry[f"{model_name}"],
        registry={
            weights_registry[f"{model_name}"][0]: weights_registry[f"{model_name}"][1],
            configs_registry[f"{model_name}"][0]: configs_registry[f"{model_name}"][1],
        },
    )

    model_type = configs_registry[f"{model_name}"][2]
    weights_name = weights_registry[f"{model_name}"][0]
    config_name = configs_registry[f"{model_name}"][0]

    return model, model_type, weights_name, config_name


def configure(
    model_name: str,
    confluency_est: int = 2000,
    conf_threshold: float = 0.3,
    save_dir: Optional[bool] = None
) -> dict:

    """
    Configures model parameters.
    INPUTS:
        model_name: str,
        confluency_est: int in the interval (0, 2000],
        conf_threshold: float in the intercal (0, 1)
    OUTPUTS:
        container: dict containing relevant variables for downstream inference
    """
    container = {}
    model, model_type, weights_name, config_name = get_model(model_name)
    if model_type == "yacs":
        model_type = "yacs"
        cfg = get_cfg()
        cfg.merge_from_file(model.fetch(f"{config_name}"))
        cfg.MODEL.WEIGHTS = model.fetch(f"{weights_name}")

        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
        else:
            cfg.MODEL.DEVICE = "cpu"

        if 0 < confluency_est <= 2000:
            cfg.TEST.DETECTIONS_PER_IMAGE = (
                confluency_est
            )
        if 0 < conf_threshold < 1:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
                conf_threshold
            )
        predictor = DefaultPredictor(cfg)

    else:
        model_type = "lazy"
        cfg = LazyConfig.load(model.fetch(f"{config_name}"))
        cfg.train.init_checkpoint = model.fetch(f"{weights_name}")

        if torch.cuda.is_available():
            cfg.train.device = "cuda"
        else:
            cfg.train.device = "cpu"

        if 0 < confluency_est <= 2000:
            cfg.model.proposal_generator.post_nms_topk[1] = (
               confluency_est
            )

        if 0 < conf_threshold < 1:
            cfg.model.roi_heads.box_predictor.test_score_thresh = (
                conf_threshold
            )

        predictor = instantiate(cfg.model)
        predictor.to(cfg.train.device)
        predictor = create_ddp_model(predictor)
        DetectionCheckpointer(predictor).load(cfg.train.init_checkpoint)
        predictor.eval()

    if save_dir == None:
        save_dir = os.getcwd()

    print("Configurations successfully saved")
    configured = True
    container.update(
        {
            "predictor" : predictor,
            "configured": configured,
            "model_type": model_type,
            "model_name": model_name,
            "confluency_est": confluency_est,
            "conf_threshold": conf_threshold,
            "save_dir": save_dir

        }
    )

    return container


def inference(
    container: dict,
    img: np.ndarray,
    frame_num: Optional[int] = None,
    analyze: Optional[bool] = False,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    """
    Runs the actual inference -> Detectron2 -> masks
    ------------------------------------------------
    INPUTS:
        container: dict, surogate object for cell_aap_widget,
        img: np.ndarray, image to run inference on,
        frame_num: int, frame number to keep track of cenroids,
        analyze, bool, whether or not to analyze results
    OUTPUTS:
        seg_fordisp: np.ndarray, semantic segmentation,
        sef_fortracking: np.ndarray, instance segmentation
        centroids: np.ndarray,
        img: list[np.ndarray], original image,
        confidence: np.ndarray
    """

    if container['model_type'] == "yacs":
        output = container['predictor'](img.astype("float32"))

    else:
        if img.shape == (2048, 2048):
            img = au.square_reshape(img, (1024, 1024))
        img_perm = np.moveaxis(img, -1, 0)

        with torch.inference_mode():
            output = container['predictor'](
                [{"image": torch.from_numpy(img_perm).type(torch.float32)}]
            )[0]

    segmentations = output["instances"].pred_masks.to("cpu")
    labels = output["instances"].pred_classes.to("cpu").numpy()
    scores = output["instances"].scores.to("cpu").numpy()
    scores = (scores*100).astype('uint16')
    classes = output['instances'].pred_classes.to("cpu").numpy()

    seg_fordisp = color_masks(
        segmentations, labels, method="custom", custom_dict={0: 1, 1: 100}
    )

    scores_mov = color_masks(segmentations, scores, method="straight")

    if analyze:
        seg_fortracking = color_masks(segmentations, labels, method="random", erode = True)
    else:
        seg_fortracking = color_masks(segmentations, labels, method="random")

    centroids = []
    for i, _ in enumerate(labels):
        labeled_mask = skimage.measure.label(segmentations[i])
        centroid = skimage.measure.centroid(labeled_mask)
        if frame_num != None:
            centroid = np.array([frame_num, centroid[0], centroid[1]])

        centroids.append(centroid)

    return seg_fordisp, seg_fortracking, centroids, img, scores_mov, classes


def run_inference(container: dict, movie_file: str, interval: list[int]):
    """
    Runs inference on image returned by self.image_select(), saves inference result if save selector has been checked
    ----------------------------------------------------------------------------------------------------------------
    INPUTS:
        container: dict, surrogate object for cell_aap_widget,
        movie_file: str, path to movie to run inference on,
        interval: list[int], range of images within movie to run inference on, for example, if the movie contains 89 images [0, 88] is the largest possible interval.
    OUTPUTS:
        result: dict containing relevant inference outputs
    """
    prog_count = 0
    instance_movie = []
    semantic_movie = []
    scores_movie = []
    classes_list = []
    points = ()


    name, im_array = str(movie_file), tiff.imread(movie_file)
    name = name.replace(".", "/").split("/")[-2]

    try:
        assert container['configured'] == True
    except AssertionError:
        raise Exception(
            "You must configure the model before running inference"
        )

    try:
       assert interval[0] >=0
       assert interval[1]<= max(im_array.shape)
    except AssertionError:
        if interval[0] <= 0:
            interval[0] = 0
        if interval[1] >= max(im_array.shape):
            interval[1] = max(im_array.shape)

    if len(im_array.shape) == 3:
        movie = []
        for frame in range(
            interval[1]
            - interval[0]
            + 1
        ):
            prog_count += 1
            frame += interval[0]
            img = au.bw_to_rgb(im_array[frame])
            semantic_seg, instance_seg, centroids, img, scores_mov, classes= inference(
                container, img, frame - interval[0]
            )
            movie.append(img)
            semantic_movie.append(semantic_seg.astype("uint16"))
            instance_movie.append(instance_seg.astype("uint16"))
            scores_movie.append(scores_mov.astype("uint16"))
            classes_list.append(classes)
            if len(centroids) != 0:
                points += (centroids,)

    elif len(im_array.shape) == 2:
        prog_count += 1
        img = au.bw_to_rgb(im_array)
        semantic_seg, instance_seg, centroids, img, scores_mov, classes= inference(container, img)
        semantic_movie.append(semantic_seg.astype("uint16"))
        instance_movie.append(instance_seg.astype("uint16"))
        scores_movie.append(scores_mov.astype("uint16"))
        classes_list.append(classes)
        if len(centroids) != 0:
            points += (centroids,)

    model_name = container['model_name']

    semantic_movie = np.asarray(semantic_movie)
    instance_movie = np.asarray(instance_movie)
    scores_movie = np.asarray(scores_movie) 
    points_array = np.vstack(points)
    classes_array = np.concatenate(classes_list, axis =0)

    cache_entry_name = f"{name}_{model_name}_{container['confluency_est']}_{round(container['conf_threshold'], ndigits = 2)}"


    result = {
            "name": cache_entry_name,
            "semantic_movie": semantic_movie,
            "instance_movie": instance_movie,
            "centroids": points_array,
            "scores_movie": scores_movie,
            "classes": classes_array
        }

    return result

def save(container, result):
    """
    Saves and analyzes an inference result
    """


    filepath = container['save_dir']
    inference_result_name = result['name']

    model_name = container["model_name"]
    try:
        position = re.search(r"_s\d_", inference_result_name).group()
        analysis_file_prefix = inference_result_name.split(position)[0] + position
    except Exception as error:
        analysis_file_prefix = inference_result_name.split(model_name)[0]


    inference_folder_path = os.path.join(filepath, inference_result_name + "_inference")
    try:
        os.mkdir(inference_folder_path)
    except OSError as error:
        print("Directory was already present, saving in found directory")

    '''
    scores = result['scores']
    classes = result['classes']
    confidence = np.asarray([scores, classes])
    confidence_df = pd.DataFrame(confidence.T, columns = ['scores', 'classes'])
    confidence_df.to_excel(
        os.path.join(inference_folder_path, analysis_file_prefix + "confidence.xlsx"), sheet_name = "confidence"
    )
    '''

    tiff.imwrite(
        os.path.join(
            inference_folder_path, analysis_file_prefix + "semantic_movie.tif"
        ),
        result["semantic_movie"],
        dtype="uint16",
    )

    tiff.imwrite(
        os.path.join(
            inference_folder_path, analysis_file_prefix + "instance_movie.tif"
        ),
        result["instance_movie"],
        dtype="uint16",
    )

    tiff.imwrite(
        os.path.join(
            inference_folder_path, analysis_file_prefix + "scores_movie.tif"
        ),
        result["scores_movie"],
    )