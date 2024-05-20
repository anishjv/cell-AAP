import napari
import torch
import sys
sys.path.append("/Users/whoisv/cell-AAP/cell_AAP/")
import numpy as np
import os
import re
import cv2
import pathlib
import tifffile as tiff
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from data_module import annotation_utils as au   # type: ignore
from skimage.measure import label
from skimage.morphology import erosion, disk
from magicclass import magicclass, field, MagicTemplate
from magicclass.widgets import PushButton, CheckBox, FloatSlider, FileEdit, ProgressBar
from magicclass.utils import thread_worker


setup_logger()
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
model = "/Users/whoisv/cell-AAP/models/20240513_171100_0.33x"
model_version = "final"
cfg = get_cfg()
cfg.merge_from_file(model + "/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join(model, f"model_{model_version}.pth")
cfg.MODEL.DEVICE = "cpu"
cfg.TEST.DETECTIONS_PER_IMAGE = 100000000000


@magicclass
class CCSN_GUI(MagicTemplate):
    """
    Napari GUI for CCSN instance segmentation algorithim
    -----------------------------------------------------
    """

    def __init__(self, viewer: napari.Viewer, predictor):
        super().__init__()
        self.viewer = viewer
        self.predictor = predictor


    @magicclass(layout="vertical")
    class Field_2:
        image_selector = field(FileEdit, name="Select Image")
        path_selector = field(FileEdit, name="Save Inference at")
        save_box = field(CheckBox, name="Save Inference")

    @magicclass
    class Field_1:
        threshold_slider = field(
            FloatSlider, name="Confidence Percentage", options={"min": 0, "max": 1}
        )
        display_button = field(PushButton, name="Display")
        inference_button = field(PushButton, name="Inference")

    @magicclass
    class Field_3:
        pbar = field(ProgressBar, name = 'Progress Bar')


    @Field_1.inference_button.connect
    @Field_3.pbar.connect
    @thread_worker(progress = {'pbar' :Field_3.pbar}, force_async=True)
    def _inference(self):
        """
        Runs inference on image returned by self._image_select(), saves inference result if self._save_select() == True
        ----------------------------------------------------------------------------------------------------------------
        """

        _, im_array = self._image_select()
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.Field_1.threshold_slider.value
        if len(im_array.shape) == 3:
            mask_array = np.empty(
                (
                    im_array.shape[0],
                    int(im_array.shape[1]),
                    int(im_array.shape[2]),
                ),
                dtype="int32",
            )

            for frame in range(im_array.shape[0]):
                img = au.bw_to_rgb(im_array[frame].astype("float32"))
                output = self.predictor(img)
                segmentations = np.asarray(output["instances"].pred_masks.to("cpu"))
                segmentations = np.bitwise_or.reduce(segmentations, axis = 0)
                segmentations = label(segmentations)
                mask_array[frame] = segmentations

        elif len(im_array.shape) == 2:
            img = au.bw_to_rgb(im_array.astype("float32"))
            output = self.predictor(img)
            segmentations = np.asarray(output["instances"].pred_masks.to("cpu"))
            segmentations = np.bitwise_or.reduce(segmentations, axis = 0)
            segmentations = label(segmentations)
            mask_array = segmentations


        save_mask, filepath = self._save_select()
        name, _ = self._image_select()
        name = name.replace(".", "/").split("/")[-2]
        if save_mask == True:
            if filepath == pathlib.Path("default/path"):
                filepath = os.getcwd()
            tiff.imwrite(
                os.path.join(filepath, f"{model}_{model_version}_{name}_.tif"),
                mask_array.astype("uint8"),
                compression="jpeg",
                compressionargs={"level": 20},
            )

        return mask_array, name
    

    @_inference.returned.connect
    def _view_inference(self, arr_name_tup):
        mask_array, name = arr_name_tup
        self.viewer.add_labels(mask_array, name=f"{name}_inference")

    def _save_select(self):
        """
        Returns the truth value of the save box selector and the path for which the inference result should be saved to
        ---------------------------------------------------------------------------------------------------------------
        """
        return self.Field_2.save_box.value, self.Field_2.path_selector.value

    def _image_select(self):
        """
        Returns the path selected in the image selector box and the array corresponding the to path
        -------------------------------------------------------------------------------------------
        """
        if (
            re.search(
                r"^.+\.(?:(?:[tT][iI][fF][fF]?)|(?:[tT][iI][fF]))$",
                str(self.Field_2.image_selector.value),
            )
            == None
        ):
            layer_data = cv2.imread(
                                    str(self.Field_2.image_selector.value), 
                                    cv2.IMREAD_GRAYSCALE
                                    )
        else:
            layer_data = tiff.imread(
                                    self.Field_2.image_selector.value
                                    )


        return str(self.Field_2.image_selector.value), layer_data
    
    @Field_1.display_button.connect
    def _view_file(self):
        """
        Displays file in Napari gui if file has been selected, also returns the 'name' of the image file
        ------------------------------------------------------------------------------------------------
        """
        if self._image_select():
            name, layer_data = self._image_select()
            name = name.replace(".", "/").split("/")[-2]
            viewer.add_image(layer_data, name=name)
        else:
            raise ValueError("No image has been selected")
    


if __name__ == "__main__":
    viewer = napari.Viewer()
    predictor = DefaultPredictor(cfg)
    viewer.window.add_dock_widget(CCSN_GUI(viewer, predictor), area="right")
    napari.run()
