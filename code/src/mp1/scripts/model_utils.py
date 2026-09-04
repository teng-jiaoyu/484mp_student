import torch
import numpy as np
import cv2
from simple_enet import SimpleENet


##### YOUR CODE STARTS HERE #####
# DO NOT CHANGE ANY FUNCTION HEADERS

# load your best model
def load_model() -> SimpleENet:
    path_to_your_model = "data/FILL_THIS_OUT"
    model = SimpleENet()
    model.load_state_dict(torch.load(path_to_your_model, weights_only=True))
    return model

def inference(model: SimpleENet, image: np.ndarray, device: str) -> np.ndarray:
    """
    The main image processing pipeline for your model
    
    :param model: pytorch model
    :type model: SimpleENet
    :param image: a BGR image taken from the GEM's camera
    :type image: np.ndarray
    :param device: the device on which the model should run on ("cpu" or "cuda")
    :type device: str
    :return: binary lane-segmented image
    :rtype: ndarray
    """
    pred = None
    return pred


##### YOUR CODE ENDS HERE #####