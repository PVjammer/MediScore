"""
* Date: 5/25/2018
* Authors: Yooyoung Lee and Timothee Kheyrkhah

* Description: this code calculates performance measures (for points, AUC, and EER)
on system outputs (confidence scores) and return report plot(s) and table(s).

* Disclaimer:
This software was developed at the National Institute of Standards
and Technology (NIST) by employees of the Federal Government in the
course of their official duties. Pursuant to Title 17 Section 105
of the United States Code, this software is not subject to copyright
protection and is in the public domain. NIST assumes no responsibility
whatsoever for use by other parties of its source code or open source
server, and makes no guarantees, expressed or implied, about its quality,
reliability, or any other characteristic."

"""

import argparse
import numpy as np
import pandas as pd
import logging
import os  # os.system("pause") for windows command line
import sys

from collections import OrderedDict
from itertools import cycle

from detection import *
from DetectiionScorerCLI import command_interface

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
import Render as p
import detMetrics as dm
import Partition as f

logging.basicConfig(level=logging.INFO)




if __name__ == '__main__':

    if len(sys.argv) == 1:
        class ArgsList():
            def __init__(self):
                print("Debugging mode: initiating ...")
                # Inputs
                self.task = "manipulation"
                self.refDir = "../../data/test_suite/detectionScorerTests/reference"
                self.inRef = "NC2017-manipulation-ref.csv"
                self.inIndex = "NC2017-manipulation-index.csv"
                self.sysDir = "../../data/test_suite/detectionScorerTests/baseline"
                self.inSys = "Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
                # TSV
                #self.tsv = "tsv_example/q-query-example.tsv"
                self.tsv = ""
                # Metrics
                self.farStop = 0.05
                self.ci = False
                self.ciLevel = 0.9
                self.dLevel = 0.0
                # Outputs
                self.outRoot = "./testcase/test"
                self.outMeta = False
                self.outSubMeta = False
                self.dump = False
                self.verbose = False
                #  Plot options
                self.plotTitle = "Performance"
                self.plotSubtitle = "bla"
                self.plotType = "roc"
                self.display = True
                self.multiFigs = False
                self.configPlot = ""
                self.noNum = False
                # Query options
                self.query = ""
                self.queryPartition = ""
                self.queryManipulation = ["TaskID==['manipulation']"]
                #self.queryManipulation = ""
                self.optOut = False
                self.verbose = True

        args = ArgsList()
    else:
        args = command_interface()

    score(args)
