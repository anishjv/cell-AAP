import numpy as np
import re
from typing import Optional
import tifffile as tiff
import os
import cell_AAP.core.inference_core as inference_core  # type: ignore


# Use shared functions from core module


def configure(
    model_name: str,
    confluency_est: int = 2000,
    conf_threshold: float = 0.3,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Configures model parameters and returns a container dict
    -------------------------------------------------------
    INPUTS:
        model_name: str
            Name of the model to configure
        confluency_est: int
            Maximum detections per image in the interval (0, 2000]
        conf_threshold: float
            Confidence threshold in the interval (0, 1)
        save_dir: Optional[str]
            Directory to save results (defaults to current working directory)
    OUTPUTS:
        container: dict
            Dictionary containing relevant variables for downstream inference
    """
    predictor, model_type, cfg = inference_core.configure_predictor(
        model_name, confluency_est, conf_threshold
    )

    if save_dir is None:
        save_dir = os.getcwd()

    print("Configurations successfully saved")
    container = {
        "predictor": predictor,
        "configured": True,
        "model_type": model_type,
        "model_name": model_name,
        "confluency_est": confluency_est,
        "conf_threshold": conf_threshold,
        "save_dir": save_dir,
    }

    return container


def inference(
    container: dict,
    img: np.ndarray,
    frame_num: Optional[int] = None,
    keep_resized_output: Optional[bool] = False,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the actual inference (Detectron2) and produces masks
    --------------------------------------------------------
    INPUTS:
        container: dict
            Dictionary containing predictor, model_type, and other configuration
        img: np.ndarray
            Image to run inference on
        frame_num: Optional[int]
            Frame number to keep track of centroids
        keep_resized_output: Optional[bool]
            If True, returns 1024x1024 output. If False (default), returns output matching input size.
    OUTPUTS:
        seg_fordisp: np.ndarray
            Semantic segmentation for display
        seg_fortracking: np.ndarray
            Instance segmentation for tracking
        centroids: list
            List of centroid coordinates
        img: np.ndarray
            Image (original or resized depending on keep_resized_output)
        scores_mov: np.ndarray
            Confidence scores as mask image
        classes: np.ndarray
            Class labels for each detection
    """
    return inference_core.run_inference_on_image(
        container["predictor"],
        container["model_type"],
        img,
        frame_num,
        keep_resized_output,
    )


def run_inference(
    container: dict,
    movie_file: str,
    interval: list[int],
    keep_resized_output: Optional[bool] = False,
):
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
        assert container["configured"] == True
    except AssertionError:
        raise Exception("You must configure the model before running inference")

    try:
        assert interval[0] >= 0
        assert interval[1] <= max(im_array.shape)
    except AssertionError:
        if interval[0] <= 0:
            interval[0] = 0
        if interval[1] >= max(im_array.shape):
            interval[1] = max(im_array.shape)

    if len(im_array.shape) == 3:
        movie = []
        for frame in range(interval[1] - interval[0] + 1):
            frame += interval[0]
            img = im_array[frame]
            semantic_seg, instance_seg, centroids, img, scores_mov, classes = inference(
                container,
                img,
                frame - interval[0],
                keep_resized_output=keep_resized_output,
            )
            movie.append(img)
            semantic_movie.append(semantic_seg.astype("uint16"))
            instance_movie.append(instance_seg.astype("uint16"))
            scores_movie.append(scores_mov.astype("uint16"))
            classes_list.append(classes)
            if len(centroids) != 0:
                points += (centroids,)
            prog_count += 1

    elif len(im_array.shape) == 2:
        semantic_seg, instance_seg, centroids, img, scores_mov, classes = inference(
            container, im_array, keep_resized_output=keep_resized_output
        )
        semantic_movie.append(semantic_seg.astype("uint16"))
        instance_movie.append(instance_seg.astype("uint16"))
        scores_movie.append(scores_mov.astype("uint16"))
        classes_list.append(classes)
        if len(centroids) != 0:
            points += (centroids,)
        prog_count += 1

    model_name = container["model_name"]

    semantic_movie = np.asarray(semantic_movie)
    instance_movie = np.asarray(instance_movie)
    scores_movie = np.asarray(scores_movie)
    points_array = np.vstack(points)
    classes_array = np.concatenate(classes_list, axis=0)

    cache_entry_name = f"{name}_{model_name}_{container['confluency_est']}_{round(container['conf_threshold'], ndigits = 2)}"

    result = {
        "name": cache_entry_name,
        "semantic_movie": semantic_movie,
        "instance_movie": instance_movie,
        "centroids": points_array,
        "scores_movie": scores_movie,
        "classes": classes_array,
    }

    return result


def save(container, result):
    """
    Saves and analyzes an inference result
    """

    filepath = container["save_dir"]
    inference_result_name = result["name"]

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

    """
    scores = result['scores']
    classes = result['classes']
    confidence = np.asarray([scores, classes])
    confidence_df = pd.DataFrame(confidence.T, columns = ['scores', 'classes'])
    confidence_df.to_excel(
        os.path.join(inference_folder_path, analysis_file_prefix + "confidence.xlsx"), sheet_name = "confidence"
    )
    """

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
        os.path.join(inference_folder_path, analysis_file_prefix + "scores_movie.tif"),
        result["scores_movie"],
    )
