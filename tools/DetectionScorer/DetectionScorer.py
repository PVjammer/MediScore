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

    # the performers' result directory
    if '/' not in args.outRoot:
        root_path = '.'
        file_suffix = args.outRoot
    else:
        root_path, file_suffix = args.outRoot.rsplit('/', 1)

    if root_path != '.' and not os.path.exists(root_path):
        os.makedirs(root_path)

    if (not args.query) and (not args.queryPartition) and (not args.queryManipulation) and (args.multiFigs is True):
        logger.error("ERROR: The multiFigs option is not available without query options.")
        exit(1)

    # this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
    # TODO Can I just do a fillna on the df to get rid of nans?
    sys_dtype = {'ConfidenceScore': str}

    #TODO Do we use TSV files?
    if args.tsv:
        print("Place TSV metrics here ...")
        DM_List, table_df = None, None
        index_m_df = load_csv(os.path.join(args.refDir, args.inRef),
                              mysep='\t', mydtype=sys_dtype)
    else:
        """
        index_m_df: Pandas dataframe containing the reference index and the analytic system output
        sys_ref_overlap: List of overlapping columns in the system output and reference datframes
        input_ref_idx_sys:
            - refDir:     Reference and index data path
            - inRef:      Reference csv file that contains the ground-truth and metadata info [e.g., references/ref.csv]', metavar='character'
            - inIndex:    CSV index file
            - sysDir:     System output data path
            - inSys:      CSV file of the system output (i.e. Analytic results)
            - outRoot:    report path and the file name prefix for saving the plot(s) and table (s)
            - outSubMeta: Boolean: Should the system save a csv file with the system output with minimal metadata?
            - sys_dtype:  Data type of the system output (I think)
        """
        index_m_df, sys_ref_overlap = input_ref_idx_sys(args.refDir, args.inRef, args.inIndex, args.sysDir,
                                                        args.inSys, args.outRoot, args.outSubMeta, sys_dtype)

    # Total number of entries in the dataframe
    total_num = index_m_df.shape[0]
    logging.info("Total data number: {}".format(total_num))

    sys_response = 'all'  # to distinguish use of the optout
    query_mode = ""
    tag_state = '_all'

    if args.optOut:
        """
        Get truncated merged dataframe based on optout.
        Does this just replace the dataframe returned from input_ref_idx_sys??
        """
        sys_response = 'tr'
        index_m_df = is_optout(index_m_df, sys_response)



    # Query Mode
    elif args.query or args.queryPartition or args.queryManipulation:
        if args.query:
            query_mode = 'q'
            query_str = args.query
        elif args.queryPartition:
            query_mode = 'qp'
            query_str = args.queryPartition
        elif args.queryManipulation:
            query_mode = 'qm'
            query_str = args.queryManipulation

        tag_state = '_' + query_mode + '_query'

        logging.info("Query_mode: {}, Query_str: {}".format(query_mode,query_str))
        DM_List, table_df, selection = yes_query_mode(index_m_df, args.task, args.refDir, args.inRef, args.outRoot,
                                                      args.optOut, args.outMeta, args.farStop, args.ci, args.ciLevel, args.dLevel, total_num, sys_response, query_str, query_mode, sys_ref_overlap)
        # Render plots with the options
        q_opts_list, q_plot_opts = plot_options(DM_List, args.configPlot, args.plotType,
                                                args.plotTitle, args.plotSubtitle, args.optOut)
        opts_list, plot_opts = query_plot_options(
            DM_List, q_opts_list, q_plot_opts, selection, args.optOut, args.noNum)

    # No Query mode
    else:
        #print(index_m_df.columns)
        DM_List, table_df = no_query_mode(index_m_df, args.task, args.refDir, args.inRef, args.outRoot,
                                          args.optOut, args.outMeta, args.farStop, args.ci, args.ciLevel, args.dLevel, total_num, sys_response)
        # Render plots with the options
        opts_list, plot_opts = plot_options(DM_List, args.configPlot, args.plotType,
                                            args.plotTitle, args.plotSubtitle, args.optOut)

    logging.info("Rendering/saving csv tables...\n")
    if isinstance(table_df, list):
        print("\nReport tables:\n")
        for i, table in enumerate(table_df):
            print("\nPartition {}:".format(i))
            print(table)
            table.to_csv(args.outRoot + tag_state + '_' + str(i) + '_report.csv', index=False, sep='|')
    else:
        print("Report table:\n{}".format(table_df))
        table_df.to_csv(args.outRoot + tag_state + '_report.csv', index=False, sep='|')

    if args.dump:
        logging.info("Dumping metric objects ...\n")
        for i, DM in enumerate(DM_List):
            DM.write(root_path + '/' + file_suffix + '_query_' + str(i) + '.dm')

    logging.info("Rendering/saving plots...\n")
    # Creation of the object setRender (~DetMetricSet)
    configRender = p.setRender(DM_List, opts_list, plot_opts)
    # Creation of the Renderer
    myRender = p.Render(configRender)
    # Plotting
    myfigure = myRender.plot_curve(
        args.display, multi_fig=args.multiFigs, isOptOut=args.optOut, isNoNumber=args.noNum)

    # save multiple figures if multi_fig == True
    if isinstance(myfigure, list):
        for i, fig in enumerate(myfigure):
            fig.savefig(args.outRoot + tag_state + '_' + str(i)  + '_' + plot_opts['plot_type'] + '.pdf', bbox_inches='tight')
    else:
        myfigure.savefig(args.outRoot + tag_state + '_' + plot_opts['plot_type'] +'.pdf', bbox_inches='tight')
