import torch
import numpy as np
import cv2
from simple_enet import SimpleENet


##### YOUR CODE STARTS HERE #####
# DO NOT CHANGE ANY FUNCTION HEADERS

# load your best model
def load_model() -> SimpleENet:
    path_to_your_model = "epoch20.pth"
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
    original_height, original_width = image.shape[:2]

    resized = cv2.resize(image, (640, 384))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(normalized)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    prediction = prediction.squeeze(0)
    prediction = prediction.cpu().numpy().astype(np.uint8)

    prediction = cv2.resize(
        prediction,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    return prediction

##### YOUR CODE ENDS HERE #####
