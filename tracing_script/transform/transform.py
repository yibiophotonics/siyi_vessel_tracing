import torchvision.transforms as transforms
import random

class RotateDiscreteAngle(object):
    def __init__(self, angle):
        """
        args:
            angle: (int list): angle to rotate the image
            notice: it is a list to include which angle to rotate
        """
        self.angle = angle

    def __call__(self, sample):
        """
        args:
            sample: input image
        function:
            rotate the image with a random angle in the list
        """
        sample = transforms.functional.rotate(sample, angle= random.choice(self.angle))

        return sample