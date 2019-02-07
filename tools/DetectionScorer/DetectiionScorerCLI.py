import argparse

########### Command line interface ########################################################
def command_interface():
    def is_file_specified(x):
        if x == '':
            raise argparse.ArgumentTypeError("{0} not provided".format(x))
        return x

    def restricted_float(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
        return x

    def restricted_ci_value(x):
        if x == '':
            raise argparse.ArgumentTypeError("{0} not provided".format(x))

        x = float(x)
        if x <= 0.8 or x >= 0.99:
            raise argparse.ArgumentTypeError("%r not in range [0.80, 0.99]" % (x,))
        return x

    def restricted_dprime_level(x):
        if x == '':
            raise argparse.ArgumentTypeError("{0} not provided".format(x))
        x = float(x)
        if x > 0.3 or x < 0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 0.3]" % (x,))
        return x

    parser = argparse.ArgumentParser(description='NIST detection scorer.')
    # Task Type Options
    parser.add_argument('-t', '--task', default='manipulation',
                        choices=['manipulation', 'splice', 'eventverification', 'camera'],
                        help='Define the target task for evaluation(default: %(default)s)', metavar='character')
    # Input Options
    parser.add_argument('--refDir', default='.',
                        help='Specify the reference and index data path: [e.g., ../NC2016_Test] (default: %(default)s)', metavar='character')
    parser.add_argument('-tv', '--tsv', default='',
                        help='Specify the reference TSV file that contains the ground-truth and metadata info [e.g., results.tsv]', metavar='character')
    parser.add_argument('-r', '--inRef', default='', type=is_file_specified,
                        help='Specify the reference CSV file that contains the ground-truth and metadata info [e.g., references/ref.csv]', metavar='character')
    parser.add_argument('-x', '--inIndex', default='', type=is_file_specified,
                        help='Specify the index CSV file: [e.g., indexes/index.csv] (default: %(default)s)', metavar='character')
    parser.add_argument('--sysDir', default='.',
                        help='Specify the system output data path: [e.g., /mySysOutputs] (default: %(default)s)', metavar='character')  # Optional
    parser.add_argument('-s', '--inSys', default='', type=is_file_specified,
                        help='Specify a CSV file of the system output formatted according to the specification: [e.g., expid/system_output.csv] (default: %(default)s)', metavar='character')
    # Metric Options
    parser.add_argument('--farStop', type=restricted_float, default=0.1,
                        help='Specify the stop point of FAR for calculating partial AUC, range [0,1] (default: %(default) FAR 10%)', metavar='float')
    # TODO: relation between ci and ciLevel
    parser.add_argument('--ci', action='store_true',
                        help="Calculate the lower and upper confidence interval for AUC if this option is specified. The option will slowdown the speed due to the bootstrapping method.")
    parser.add_argument('--ciLevel', type=restricted_ci_value, default=0.9,
                        help="Calculate the lower and upper confidence interval with the specified confidence level, The option will slowdown the speed due to the bootstrapping method.", metavar='float')
    parser.add_argument('--dLevel', type=restricted_dprime_level, default=0.0,
                        help="Define the lower and upper exclusions for d-prime calculation", metavar='float')
    # Output Options
    parser.add_argument('-o', '--outRoot', default='.',
                        help='Specify the report path and the file name prefix for saving the plot(s) and table (s). For example, if you specify "--outRoot test/NIST_001", you will find the plot "NIST_001_det.png" and the table "NIST_001_report.csv" in the "test" folder: [e.g., temp/xx_sys] (default: %(default)s)', metavar='character')
    parser.add_argument('--outMeta', action='store_true',
                        help="Save a CSV file with the system output with metadata")
    parser.add_argument('--outSubMeta', action='store_true',
                        help="Save a CSV file with the system output with minimal metadata")
    parser.add_argument('--dump', action='store_true',
                        help="Save the dump files (formatted as a binary) that contains a list of FAR, FPR, TPR, threshold, AUC, and EER values. The purpose of the dump files is to load the point values for further analysis without calculating the values again.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print output with procedure messages on the command-line if this option is specified.")
    # Plot Options
    parser.add_argument('--plotTitle', default='Performance',
                        help="Define a plot title (default: %(default)s)", metavar='character')
    parser.add_argument('--plotSubtitle', default='',
                        help="Define a plot subtitle (default: %(default)s)", metavar='character')
    parser.add_argument('--plotType', default='', choices=['roc', 'det'],
                        help="Define a plot type:[roc] and [det] (default: %(default)s)", metavar='character')
    parser.add_argument('--display', action='store_true',
                        help="Display a window with the plot(s) on the command-line if this option is specified.")
    parser.add_argument('--multiFigs', action='store_true',
                        help="Generate plots(with only one curve) per a partition ")
    parser.add_argument('--noNum', action='store_true',
                        help="Do not print the number of target trials and non-target trials on the legend of the plot")
    # Custom Plot Options
    parser.add_argument('--configPlot', default='',
                        help="Load a JSON file that allows user to customize the plot (e.g. change the title font size) by augmenting the json files located in the 'plotJsonFiles' folder.")
    # Performance Evaluation by Query Options
    factor_group = parser.add_mutually_exclusive_group()

    factor_group.add_argument('-q', '--query', nargs='*',
                              help="Evaluate system performance on a partitioned dataset (or subset) using multiple queries. Depending on the number (N) of queries, the option generates N report tables (CSV) and one plot (PDF) that contains N curves.", metavar='character')
    factor_group.add_argument('-qp', '--queryPartition',
                              help="Evaluate system performance on a partitioned dataset (or subset) using one query. Depending on the number (M) of partitions provided by the cartesian product on query conditions, this option generates a single report table (CSV) that contains M partition results and one plot that contains M curves. (syntax retriction: '==[]','<','<=')", metavar='character')
    factor_group.add_argument('-qm', '--queryManipulation', nargs='*',
                              help="This option is similar to the '-q' option; however, the queries are only applied to the target trials (IsTarget == 'Y') and use all of non-target trials. Depending on the number (N) of queries, the option generates N report tables (CSV) and one plot (PDF) that contains N curves.", metavar='character')
    parser.add_argument('--optOut', action='store_true',
                        help="Evaluate system performance on trials where the IsOptOut value is 'N' only or the ProbeStatus values are ['Processed', 'NonProcessed', 'OptOutLocalization', 'FailedValidation']")

    args = parser.parse_args()

    return args
