
import scoreservice
import sys

import scoring_pb2 as pb
from tools.DetectionScorer.detection import *

def score_detection(req, resp):
    class ArgsList():
        def __init__(self, req):
            print("Debugging mode: initiating ...")
            # Inputs
            self.task = req.task
            self.refDir = req.dataset.reference_dir
            self.inRef = req.dataset.reference_gt   #"NC2017-manipulation-ref.csv"
            self.inIndex = req.dataset.index        #"NC2017-manipulation-index.csv"
            self.sysDir = req.results.results_dir   #"../../data/test_suite/detectionScorerTests/baseline"
            self.inSys = req.results.results_file   #"Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
            # TSV
            #self.tsv = "tsv_example/q-query-example.tsv"
            self.tsv = ""
            # Metrics
            self.farStop = req.options.farstop      # 0.05
            self.ci = req.options.ci                # False
            self.ciLevel = req.options.ciLevel      # 0.9
            self.dLevel = req.options.dLevel        # 0.0
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
            self.query = req.results.query
            self.queryPartition = ""
            self.queryManipulation = ["TaskID==['manipulation']"]
            #self.queryManipulation = ""
            self.optOut = req.results.optout


    print("Running Function 'Score Detection'")
    args = ArgsList(req)
    score(args)
    resp.output_dir = args.outRoot

    logging.info("Done")


def main():
     svc = scoreservice.ScoringService()
     svc.RegisterDetection(score_detection)
     sys.exit(svc.Run())

if __name__ == "__main__":
    main()
