import pooch
import numpy as np
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



def get_model(model_name: str):
    """
    Instaniates POOCH instance containing model files from the model_registry
    --------------------------------------------------------------------------
    INPUTS:
        cellaap_widget: instance of ui.cellAAPWidget()I
    """

    models = ['HeLa', 'U2OS']
    try:
        assert model_name in models
    except AssertionError:
        raise Exception(f'Invalid model name selected, model must be one of {models}')

    url_registry = {
        "HeLa": "doi:10.5281/zenodo.14226948",
        "U2OS": "doi:10.5281/zenodo.14226985"
    }

    weights_registry = {
        "HeLa": (
            "model_0043499.pth",
            "md5:ad056dc159ea8fd12f7d5d4562c368a9",
        ),
        "U2OS": (
            "model_0030449.pth",
            "md5:4d65600b92560d7fcda6c6fd59fa0fe8"
        )
    }

    configs_registry = {
        "HeLa": (
            "config.yaml",
            "md5:319ec68250d7ae499a274f7c4f151513",
            "lazy",
        ),
        "U2OS" : (
            "config.yaml",
            "md5:ad80d579860c53a84ab076c4db2604fd",
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
    conf_threshold: float = 0.3
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

    print("Configurations successfully saved")
    configured = True
    container.update(
        {
            "predictor" : predictor,
            "configured": configured,
            "model_type": model_type,
            "model_name": model_name,
            "confluency_est": confluency_est,
            "conf_threshold": conf_threshold
        }
    )

    return container


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
        labels: list
        method: str
        custom_dict: dict
    OUTPUTS:
        seg_labeled: np.ndarray
    """

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
            if method == "custom" and custom_dict != None:
                for j in custom_dict.keys():
                    if labels[i] == j:
                        seg_labeled[mask] += custom_dict[j]

            else:
                if erode == True:
                    mask = binary_erosion(mask, disk(3))
                if labels[i] == 0:
                    seg_labeled[mask] = 2 * i
                else:
                    seg_labeled[mask] = 2 * i - 1

    return seg_labeled


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
        if img.shape != (2048, 2048):
            img = au.square_reshape(img, (2048, 2048))
        output = container['predictor'](img.astype("float32"))

    else:
        if img.shape != (1024, 1024):
            img = au.square_reshape(img, (1024, 1024))
        img_perm = np.moveaxis(img, -1, 0)

        with torch.inference_mode():
            output = container['predictor'](
                [{"image": torch.from_numpy(img_perm).type(torch.float32)}]
            )[0]

    segmentations = output["instances"].pred_masks.to("cpu")
    labels = output["instances"].pred_classes.to("cpu")
    scores = output["instances"].scores.to("cpu").numpy()
    classes = output['instances'].pred_classes.to("cpu").numpy()
    confidence = np.vstack(
        (scores, classes)
    )

    seg_fordisp = color_masks(
        segmentations, labels, method="custom", custom_dict={0: 1, 1: 100}
    )

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

    return seg_fordisp, seg_fortracking, centroids, img, confidence


def run_inference(container: dict, movie_file: str, interval: list[int]):
    """
    Runs inference on image returned by self.image_select(), saves inference result if save selector has been checked
    ----------------------------------------------------------------------------------------------------------------
    INPUTS:
        container: dict, surogate object for cell_aap_widget,
        movie_file: str, path to movie to run inference on,
        interval: list[int], range of images within movie to run inference on, for example, if the movie contains 89 images [0, 88] is the largest possible interval.
    OUTPUTS:
        result: dict containing relevant inference outputs
    """
    prog_count = 0
    instance_movie = []
    semantic_movie = []
    confidences = []
    points = ()


    name, im_array = str(movie_file), tiff.imread(movie_file)
    name = name.replace(".", "/").split("/")[-2]

    try:
        assert container['configured'] == True
    except AssertionError:
        raise Exception(
            "You must configure the model before running inference"
        )
        return

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
            semantic_seg, instance_seg, centroids, img, confidence= inference(
                container, img, frame - interval[0]
            )
            movie.append(img)
            semantic_movie.append(semantic_seg.astype("uint16"))
            instance_movie.append(instance_seg.astype("uint16"))
            confidences.append(confidence)
            if len(centroids) != 0:
                points += (centroids,)

    elif len(im_array.shape) == 2:
        prog_count += 1
        img = au.bw_to_rgb(im_array)
        semantic_seg, instance_seg, centroids, img, confidence= inference(container, img)
        semantic_movie.append(semantic_seg.astype("uint16"))
        instance_movie.append(instance_seg.astype("uint16"))
        confidences.append(confidence)
        if len(centroids) != 0:
            points += (centroids,)

    model_name = container['model_name']

    semantic_movie = np.asarray(semantic_movie)
    instance_movie = np.asarray(instance_movie)
    points_array = np.vstack(points)

    cache_entry_name = f"{name}_{model_name}_{container['confluency_est']}_{round(container['conf_threshold'], ndigits = 2)}"


    result = {
            "name": cache_entry_name,
            "semantic_movie": semantic_movie,
            "instance_movie": instance_movie,
            "centroids": points_array,
            "confidence": confidences
        }

    return result
