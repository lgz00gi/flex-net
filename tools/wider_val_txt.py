#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matlab.engine
import os




if __name__ == '__main__':

    step_size = 10000
    iters = 10
    print iters
    output_main = '/home/guizi/project/face/wider_val/flex_guizi/'
    matlab_tool_folder = '/home/guizi/project/face/eval_tools/'
    ap_f = output_main + 'ap.txt'
    ap_w = open(ap_f, 'wb')

    print 'load matlab engine'
    os.chdir(matlab_tool_folder)
    eng = matlab.engine.start_matlab()


    for index in range(iters):
        iter_str = str((index + 1) * step_size)
        output = output_main + iter_str + '/'

        print output
        ap_list = eng.winder_eval_f(output, iter_str)
        print 'iter {0} ap is {1}'.format(iter_str, str(ap_list))
        ap_w.write(iter_str + '\r\n')
        for ap in ap_list:
            ap_w.write(str(ap[0]) + '\t' )
        ap_w.write('\n')

    ap_w.close()



