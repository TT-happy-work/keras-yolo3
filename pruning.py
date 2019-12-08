
import numpy as np

def get_prunable_layers():
    # prunable:
    # conv followed by a conv
    # first conv in residual

    # different handling:
    # 1. second convolution in residual block
    # 2. conv before residual block

    ######################### with original darknet 53  ######################################
    prunable_layers_id = [1, 3, 6, 8, 11,13,15,17,19,21,23,25, # until first skip
                            28, 30, 32, 34, 36, 38, 40,42,     # until second skip
                            45, 47, 49,51,                     # until the end of darknet
                            53, 54, 55, 56, 57,                # until large objs branch
                            60, 61, 62, 63, 64,65,68,69,70,71,72,73,
                            58,66,74                           # conv before objects branches -
                            # objects branches -    59,67,75
                          ]

    ######################### darknet resnet 5 blocks  ######################################
    prunable_layers_id = [3,4, 7,8 ,11,12, 15,16, 19,20,            # darknet resnet
                            22, 23, 24, 25, 26, 28,29, 30,31,32,    # out of the backbone
                            33, 34, 35, 37, 38, 39,40, 41,
                            42, 43
                          ]


    prunable_conv_layers = ['conv2d_{}'.format(v) for v in prunable_layers_id]


    return prunable_conv_layers
