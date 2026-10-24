"""Image preprocessing shared by training, evaluation and inference."""

from torchvision import transforms
import torchvision.transforms.functional as TF

INPUT_SIZE = 336                         # 336 rather than 224 keeps the corner digits legible
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToSquare:
    """Pad a PIL image to a square with black bars (centred), so resizing to the square
    network input never squashes the cards' round symbols into ovals."""

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        left = (side - w) // 2
        top = (side - h) // 2
        return TF.pad(img, (left, top, side - w - left, side - h - top), fill=self.fill, padding_mode="constant")


def eval_transform(size: int = INPUT_SIZE):
    """Pad to square, resize, normalise: what every crop goes through before the network."""
    return transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
