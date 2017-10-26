# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc

from stylize import stylize
import time
import math
from argparse import ArgumentParser





def main():
    a=time.time()



    # 默认参数
    content_weight = 5e0
    style_weight = 1e2
    tv_weight = 1e2
    learning_rate = 1e1
    style_scale = 1.0
    iterations = 1000
    VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

    # 必选参数
    content_image_path = "content.jpg"
    four_layer_style_path = "style.jpg",  # 这里的输入替换成对于4个层都不同的输入，不采用混合style
    output_path = None

    # 可选

    print_iterations = None
    ckpt_output = "output{}.jpg"
    ckpt_iterations = iterations / 20
    width = None
    style_scales = None
    initial = None
    style_blend_weights = [1.0, 1.0, 1.0, 1.0,1.0]

    #assert len(four_layer_style_path) == 5
    #assert len(style_blend_weights) == 5








    if not os.path.isfile(VGG_PATH):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imread(content_image_path)
    style_images = [imread(style) for style in four_layer_style_path]


    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape

    for i in range(len(style_images)):
        if style_scales is not None:
            style_scale = style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])

    assert ckpt_output and "{}"  in ckpt_output


    for iteration, image in stylize(
        network=VGG_PATH,
        initial=initial,
        content=content_image,
        styles=style_images,
        iterations=iterations,
        content_weight=content_weight,
        style_weight=style_weight,
        style_blend_weights=style_blend_weights,
        tv_weight=tv_weight,
        learning_rate=learning_rate,
        print_iterations=print_iterations,
        checkpoint_iterations=ckpt_iterations
    ):
        output_file = None
        if iteration is not None:
            if ckpt_output:
                output_file = ckpt_output.format(iteration)
        else:
            output_file = output_path
        if output_file:
            imsave(output_file, image)
            print(time.time()-a)


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


if __name__ == '__main__':
    main()
