# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger

class blendedImageAttack(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, target_image, blended_rate):
        #cifar\gtsrb
        self.target_image = target_image
        self.blended_rate = blended_rate
        #mnist
        # import numpy as np
        # # 将 PIL.Image 对象转换为灰度图像
        # gray_image = np.mean(target_image, axis=2)
        # self.target_image = gray_image
        
        # self.blended_rate = blended_rate

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return (1-self.blended_rate) * img + (self.blended_rate) * self.target_image