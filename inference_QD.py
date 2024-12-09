# author : Trung Thanh Nguyen(Jimmy) | 09/12/2004  | ng.trungthanh04@gmail.com
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from model_CNN_QuickDraw import CNN
def get_args():
    parser = argparse.ArgumentParser("Test Argument")
    parser.add_argument("--image-size", "-i", type=int, default=28)
    parser.add_argument("--image-path", "-p", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-t", type=str, default="./train_model_QuickDraw/quickdraw.pth")
    args = parser.parse_args()
    return args
def inference(args):
    categories = ["airplane", "apple", "axe", "bat", "book", "boomerang", "flower", "mushroom", "pencil", "sun",
                  "sword"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image[np.newaxis, np.newaxis, :, :] / 255.0
    image = torch.from_numpy(image).float()
    image = image.to(device)
    model = torch.load(args.checkpoint_path,map_location=device)
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        prediction = model(image)
        prob = softmax(prediction)
    max_value, max_index = torch.max(prob, dim=1)
    predicted_class = categories[max_index.item()]
    confidence = max_value.item()
    cv2.putText(
        ori_image,
        f"Class: {predicted_class}, Conf: {confidence:.4f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow("Prediction", ori_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    args = get_args()
    inference(args)
