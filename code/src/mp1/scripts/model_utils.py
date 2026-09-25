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

    # 1. resize
    image = cv2.resize(image, (640, 384))

    # 2. BGR -> grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. numpy -> float tensor, 0~1
    x = torch.from_numpy(image).float()[None, ...] / 255.0

    # 4. batch dimension
    # [1, 384, 640] -> [1, 1, 384, 640]
    x = x.unsqueeze(0)

    # 5. put into model's device
    x = x.to(device)

    # 6. inference：no_gradient
    with torch.no_grad():
        yp = model(x)

    # 7. choose the max score from the two classes in each pixel
    # [1, 2, 384, 640] -> [1, 384, 640]
    pred = torch.argmax(yp, dim=1)

    # 8. delete batch dimension，-> numpy
    # [1, 384, 640] -> [384, 640]
    pred = pred.squeeze(0).cpu().numpy()

    return pred


##### YOUR CODE ENDS HERE #####