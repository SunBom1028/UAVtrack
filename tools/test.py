from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'E:/siamfc-pytorch-master/pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = 'E:/siamfc-pytorch-master/OTB100'
    e = ExperimentOTB(root_dir, version=2015)
    e.run(tracker,visualize=True)
    e.report([tracker.name])
