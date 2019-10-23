import random
import numpy as np
import torch

# Maintaining a cache list and swap a random element and the input query with p=0.5
class MyImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = [None] * self.pool_size

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.images[self.num_imgs] = image
                self.num_imgs += 1
                return_images.append(image)
            else:
                # print('random swap with image pool')
                p = random.uniform(0, 1)
                if p > 0.5:
                    # print('Da')
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # print('Het')
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
           

