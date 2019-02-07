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

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
sys.path.append(lib_path)
import Render as p
import detMetrics as dm         #This does the real work
import Partition as f

logging.basicConfig(level=logging.INFO)

def load_csv(fname, mysep='|', mydtype=None):
    """
    Function to load a csv into a pandas dataframe using the specified delimiter.  Wraps the standard
    pandas 'read_csv' method and adds error handling
    """
    try:
        df = pd.read_csv(fname, sep=mysep, dtype=mydtype, low_memory=False)
        return df
    except IOError:
        #TODO Let's change this to raise an exception instead
        print("ERROR: There was an error opening the file: {}".format(fname))
        exit(1)

def save_csv(df_list, outRoot, query_mode, report_tag):
    """
    Function to save a pandas dataframe into a csv using the specified delimiter.  Wraps the standard
    pandas 'to_csv' method and adds error handling
    """
    try:
        for i, df in enumerate(df_list):
            fname = outRoot + '_' + query_mode + '_query_' + str(i) + report_tag
            df.to_csv(fname, index=False, sep='|')
    except IOError:
        #TODO Let's change this to raise an exception instead
        print("ERROR: There was an error saving the csv file: {}".format(fname))
        exit(1)

def is_optout(df, sys_response='tr'):
    """
    Function to remove trials which opted out of the localization task.
    """
    logging.info("Removing optout trials ...\n")
    if "IsOptOut" in df.columns:
        new_df = df.query(" IsOptOut==['N', 'Localization'] ")
    elif "ProbeStatus" in df.columns:
        new_df = index_m_df.query(
            " ProbeStatus==['Processed', 'NonProcessed', 'OptOutLocalization', 'FailedValidation']")
    return new_df

def overlap_cols(mySys, myRef):
    """
    Function that compares two dataframes and returns two lists: 'no_overlap' and 'overlap'
     - no_overlap: List of column names which are exclusive to the system output
     - overlap:    List of column names which are shared by both dataframes
    """
    no_overlap = [c for c in mySys.columns if c not in myRef.columns]
    overlap = [c for c in mySys.columns if c in myRef.columns]
    return no_overlap, overlap

# Loading the specified file
def define_file_name(path, ref_fname, tag_name):
    my_fname = os.path.join(path, str(ref_fname.split('.')[:-1]).strip("['']") + tag_name)
    logging.info("Specified JT file: {}".format(my_fname))
    return my_fname

def JT_merge(ref_dir, ref_fname, mainDF):
    join_fname = define_file_name(ref_dir, ref_fname, '-probejournaljoin.csv')
    mask_fname = define_file_name(ref_dir, ref_fname, '-journalmask.csv')
    if os.path.isfile(join_fname) and os.path.isfile(mask_fname):
        joinDF = pd.read_csv(join_fname, sep='|', low_memory=False)
        maskDF = pd.read_csv(mask_fname, sep='|', low_memory=False)
        jt_no_overlap, jt_overlap = overlap_cols(joinDF, maskDF)
        logging.info("JT overlap columns: {}".format(jt_overlap))
        logging.info("Merging (left join) the JournalJoin and JournalMask csv files with the reference file ...\n")
        jt_meta = pd.merge(joinDF, maskDF, how='left', on=jt_overlap)
        meta_no_overlap, meta_overlap = overlap_cols(mainDF, jt_meta)
        logging.info("JT and main_df overlap columns: {}".format(meta_overlap))
        new_df = pd.merge(mainDF, jt_meta, how='left', on=meta_overlap)
        logging.info("Main cols num: {}\n Meta cols num: {}\n, Merged cols num: {}".format(
            mainDF.shape, jt_meta.shape, new_df.shape))
        return new_df
    else:
        logging.info("JT meta files do not exist, therefore, merging process will be skipped")
        return mainDF


def input_ref_idx_sys(refDir, inRef, inIndex, sysDir, inSys, outRoot, outSubMeta, sys_dtype):
    """
    What do you do?
    """
    # Loading the reference file
    logging.info("Ref file name: {}".format(os.path.join(refDir, inRef)))
    myRefDir = os.path.dirname(os.path.join(refDir, inRef))
    myRef = load_csv(os.path.join(refDir, inRef))
    # Loading the index file
    logging.info("Index file name: {}".format(os.path.join(refDir, inIndex)))
    myIndex = load_csv(os.path.join(refDir, inIndex))
    # Loading system output
    logging.info("Sys file name: {}".format(os.path.join(sysDir, inSys)))
    mySys = load_csv(os.path.join(sysDir, inSys), mydtype=sys_dtype)

    # Identify which columns are shared between the reference csv and system output csv
    sys_ref_no_overlap, sys_ref_overlap = overlap_cols(mySys, myRef)
    logging.info("sys_ref_no_overlap: {} \n, sys_ref_overlap: {}".format(
        sys_ref_no_overlap, sys_ref_overlap))
    index_ref_no_overlap, index_ref_overlap = overlap_cols(myIndex, myRef)
    logging.info("index_ref_no_overlap: {}\n, index_ref_overlap: {}".format(
        index_ref_no_overlap, index_ref_overlap))

    # merge the reference and system output for SSD/DSD reports
    m_df = pd.merge(myRef, mySys, how='left', on=sys_ref_overlap)
    # if the confidence scores are 'nan', replace the values with the mininum score
    m_df[pd.isnull(m_df['ConfidenceScore'])] = mySys['ConfidenceScore'].min()
    # convert to the str type to the float type for computations
    m_df['ConfidenceScore'] = m_df['ConfidenceScore'].astype(np.float)

    # merge the reference and index csv (intersection only due to the partial index trials)
    index_m_df = pd.merge(m_df, myIndex, how='inner', on=index_ref_overlap)
    logging.info("index_m_df_columns: {}".format(index_m_df.columns))

    # save subset of metadata for analysis purpose
    if outSubMeta:
        logging.info("Saving the sub_meta csv file...")
        sub_pm_df = index_m_df[index_ref_overlap +
                               index_ref_no_overlap + ["IsTarget"] + sys_ref_no_overlap]
        logging.info("sub_pm_df columns: {}".format(sub_pm_df.columns))
        sub_pm_df.to_csv(outRoot + '_subset_meta.csv', index=False, sep='|')

    return index_m_df, sys_ref_overlap


def yes_query_mode(df, task, refDir, inRef, outRoot, optOut, outMeta, farStop, ci, ciLevel, dLevel, total_num, sys_response, query_str, query_mode, sys_ref_overlap):

    m_df = df.copy()
    # if the files exist, merge the JTJoin and JTMask csv files with the reference and index file
    if task in ['manipulation', 'splice', 'camera', 'eventverification']:
        logging.info("Merging the JournalJoin and JournalMask for the {} task\n".format(task))
        m_df = JT_merge(refDir, inRef, df)

    logging.info("Creating partitions for queries ...\n")
    selection = f.Partition(m_df, query_str, query_mode, fpr_stop=farStop, isCI=ci,
                            ciLevel=ciLevel, total_num=total_num, sys_res=sys_response, overlap_cols=sys_ref_overlap)
    DM_List = selection.part_dm_list
    logging.info("Number of partitions generated = {}\n".format(len(DM_List)))

    # Output the meta data as dataframe for queries
    DF_List = selection.part_df_list
    logging.info("Number of CSV partitions generated = {}\n".format(len(DF_List)))

    if outMeta:  # save all metadata for analysis purpose
        logging.info("Saving all the meta info csv file ...")
        save_csv(DF_List, outRoot, query_mode, '_allmeta.csv')

    table_df = selection.render_table()

    return DM_List, table_df, selection


def no_query_mode(df, task, refDir, inRef, outRoot, optOut, outMeta, farStop, ci, ciLevel, dLevel, total_num, sys_response):

    m_df = df.copy()

    if outMeta:  # save all metadata for analysis purpose
        logging.info("Saving all the meta info csv file ...")
        logging.info("Merging the JournalJoin and JournalMask for the {} task\n, But do not score with this data".format(task))
        meta_df = JT_merge(refDir, inRef, m_df)
        meta_df.to_csv(outRoot + '_allmeta.csv', index=False, sep='|')
        m_df.to_csv(outRoot + '_meta_scored.csv', index=False, sep='|')

    DM = dm.detMetrics(m_df['ConfidenceScore'], m_df['IsTarget'], fpr_stop=farStop,
                       isCI=ci, ciLevel=ciLevel, dLevel=dLevel, total_num=total_num, sys_res=sys_response)
    DM_List = [DM]
    table_df = DM.render_table()

    return DM_List, table_df


def plot_options(DM_list, configPlot, plotType, plotTitle, plotSubtitle, optOut):
    # Generating a default plot_options json config file
    p_json_path = "./plotJsonFiles"
    if not os.path.exists(p_json_path):
        os.makedirs(p_json_path)
    dict_plot_options_path_name = "./plotJsonFiles/plot_options.json"

    # opening of the plot_options json config file from command-line
    if configPlot:
        p.open_plot_options(dict_plot_options_path_name)

    # if plotType is indicated, then should be generated.
    if plotType == '' and os.path.isfile(dict_plot_options_path_name):
        # Loading of the plot_options json config file
        plot_opts = p.load_plot_options(dict_plot_options_path_name)
        plotType = plot_opts['plot_type']
        plot_opts['title'] = plotTitle
        plot_opts['subtitle'] = plotSubtitle
        plot_opts['subtitle_fontsize'] = 11
        #print("test plot title1 {}".format(plot_opts['title']))
    else:
        if plotType == '':
            plotType = 'roc'
        p.gen_default_plot_options(dict_plot_options_path_name, plot_title=plotTitle,
                                   plot_subtitle=plotSubtitle, plot_type=plotType.upper())
        plot_opts = p.load_plot_options(dict_plot_options_path_name)
        #print("test plot title2 {}".format(plot_opts['title']))

    # Creation of defaults plot curve options dictionnary (line style opts)
    Curve_opt = OrderedDict([('color', 'red'),
                             ('linestyle', 'solid'),
                             ('marker', '.'),
                             ('markersize', 6),
                             ('markerfacecolor', 'red'),
                             ('label', None),
                             ('antialiased', 'False')])

    # Creating the list of curves options dictionnaries (will be automatic)
    opts_list = list()
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'sienna', 'navy', 'grey',
              'darkorange', 'c', 'peru', 'y', 'pink', 'purple', 'lime', 'magenta', 'olive', 'firebrick']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    markerstyles = ['.', '+', 'x', 'd', '*', 's', 'p']
    # Give a random rainbow color to each curve
    # color = iter(cm.rainbow(np.linspace(0,1,len(DM_List)))) #YYL: error here
    color = cycle(colors)
    lty = cycle(linestyles)
    mkr = cycle(markerstyles)
    for i in range(len(DM_list)):
        new_curve_option = OrderedDict(Curve_opt)
        col = next(color)
        new_curve_option['color'] = col
        new_curve_option['marker'] = next(mkr)
        new_curve_option['markerfacecolor'] = col
        new_curve_option['linestyle'] = next(lty)
        opts_list.append(new_curve_option)

    if optOut:
        plot_opts['title'] = "tr" + plotTitle

    return opts_list, plot_opts


def query_plot_options(DM_List, opts_list, plot_opts, selection, optOut, noNum):
    # Renaming the curves for the legend
    for curve_opts, query, dm_list in zip(opts_list, selection.part_query_list, DM_List):
        trr_str = ""
        #print("plottype {}".format(plot_opts['plot_type']))
        if plot_opts['plot_type'] == 'ROC':
            met_str = " (AUC: " + str(round(dm_list.auc, 2))
        elif plot_opts['plot_type'] == 'DET':
            met_str = " (EER: " + str(round(dm_list.eer, 2))

        if optOut:
            trr_str = ", TRR: " + str(dm_list.trr)
            if plot_opts['plot_type'] == 'ROC':
                met_str = " (trAUC: " + str(round(dm_list.auc, 2))
            elif plot_opts['plot_type'] == 'DET':
                met_str = " (trEER: " + str(round(dm_list.eer, 2))

        if noNum:
            curve_opts["label"] = query + met_str + trr_str + ")"
        else:
            curve_opts["label"] = query + met_str + trr_str + ", T#: " + \
                str(dm_list.t_num) + ", NT#: " + str(dm_list.nt_num) + ")"

    return opts_list, plot_opts

def score(req):
    # the performers' result directory
    if '/' not in req.outRoot:
        root_path = '.'
        file_suffix = req.outRoot
    else:
        root_path, file_suffix = req.outRoot.rsplit('/', 1)

    if root_path != '.' and not os.path.exists(root_path):
        os.makedirs(root_path)

    if (not req.query) and (not req.queryPartition) and (not req.queryManipulation) and (req.multiFigs is True):
        logger.error("ERROR: The multiFigs option is not available without query options.")
        exit(1)

    # this should be "string" due to the "nan" value, otherwise "nan"s will have different unique numbers
    # TODO Can I just do a fillna on the df to get rid of nans?
    sys_dtype = {'ConfidenceScore': str}

    #TODO Do we use TSV files?
    if req.tsv:
        print("Place TSV metrics here ...")
        DM_List, table_df = None, None
        index_m_df = load_csv(os.path.join(req.refDir, req.inRef),
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
        index_m_df, sys_ref_overlap = input_ref_idx_sys(req.refDir, req.inRef, req.inIndex, req.sysDir,
                                                        req.inSys, req.outRoot, req.outSubMeta, sys_dtype)

    # Total number of entries in the dataframe
    total_num = index_m_df.shape[0]
    logging.info("Total data number: {}".format(total_num))

    sys_response = 'all'  # to distinguish use of the optout
    query_mode = ""
    tag_state = '_all'

    if req.optOut:
        """
        Get truncated merged dataframe based on optout.
        Does this just replace the dataframe returned from input_ref_idx_sys??
        """
        sys_response = 'tr'
        index_m_df = is_optout(index_m_df, sys_response)



    # Query Mode
    elif req.query or req.queryPartition or req.queryManipulation:
        if req.query:
            query_mode = 'q'
            query_str = req.query
        elif req.queryPartition:
            query_mode = 'qp'
            query_str = req.queryPartition
        elif req.queryManipulation:
            query_mode = 'qm'
            query_str = req.queryManipulation

        tag_state = '_' + query_mode + '_query'

        logging.info("Query_mode: {}, Query_str: {}".format(query_mode,query_str))
        DM_List, table_df, selection = yes_query_mode(index_m_df, req.task, req.refDir, req.inRef, req.outRoot,
                                                      req.optOut, req.outMeta, req.farStop, req.ci, req.ciLevel, req.dLevel, total_num, sys_response, query_str, query_mode, sys_ref_overlap)
        # Render plots with the options
        q_opts_list, q_plot_opts = plot_options(DM_List, req.configPlot, req.plotType,
                                                req.plotTitle, req.plotSubtitle, req.optOut)
        opts_list, plot_opts = query_plot_options(
            DM_List, q_opts_list, q_plot_opts, selection, req.optOut, req.noNum)

    # No Query mode
    else:
        #print(index_m_df.columns)
        DM_List, table_df = no_query_mode(index_m_df, req.task, req.refDir, req.inRef, req.outRoot,
                                          req.optOut, req.outMeta, req.farStop, req.ci, req.ciLevel, req.dLevel, total_num, sys_response)
        # Render plots with the options
        opts_list, plot_opts = plot_options(DM_List, req.configPlot, req.plotType,
                                            req.plotTitle, req.plotSubtitle, req.optOut)

    logging.info("Rendering/saving csv tables...\n")
    if isinstance(table_df, list):
        print("\nReport tables:\n")
        for i, table in enumerate(table_df):
            print("\nPartition {}:".format(i))
            print(table)
            table.to_csv(req.outRoot + tag_state + '_' + str(i) + '_report.csv', index=False, sep='|')
    else:
        print("Report table:\n{}".format(table_df))
        table_df.to_csv(req.outRoot + tag_state + '_report.csv', index=False, sep='|')

    if req.dump:
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
        req.display, multi_fig=req.multiFigs, isOptOut=req.optOut, isNoNumber=req.noNum)

    # save multiple figures if multi_fig == True
    if isinstance(myfigure, list):
        for i, fig in enumerate(myfigure):
            fig.savefig(req.outRoot + tag_state + '_' + str(i)  + '_' + plot_opts['plot_type'] + '.pdf', bbox_inches='tight')
    else:
        myfigure.savefig(req.outRoot + tag_state + '_' + plot_opts['plot_type'] +'.pdf', bbox_inches='tight')
