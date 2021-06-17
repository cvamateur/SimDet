import torch
from torch import nn
from torchvision import models
from torchsummary import summary

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class FeatureExtractor(nn.Module):
    """
    Image feature extraction using MobileNet-V2.
    """
    def __init__(self, target_height=224, target_width=224, verbose=False):
        super(FeatureExtractor, self).__init__()
        self.input_shape = (3, target_height, target_width)
        self.model = models.mobilenet_v2(pretrained=True).features
        self.model.to(device)
        if verbose:
            summary(self.model, (3, 224, 224))

    def forward(self, image):
        """
        @Params:
        -------
        image (tensor) 4-D tensor input image where the
        """
        return self.model(image)





if __name__ == '__main__':
    model = FeatureExtractor(verbose=True)