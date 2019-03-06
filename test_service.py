import grpc
import os
import scoreservice
import scoring_pb2
import scoring_pb2_grpc
import uuid

from tools.build_nist import parse_json, write_nist_csv

VALIDATION_REF_DIR = "/mnt/datasets/evaluation/MFC18_EvalPart1_Image_Ver1-Reference/reference/manipulation-image/"
VALIDATION_REF_INDEX = "MFC18_EvalPart1-manipulation-image-ref.csv"
VALIDATION_INDEX = "/mnt/datasets/evaluation/MFC19_Image_Validation_Ver1/indexes/MFC19_Validation_manipulation-image-index.csv"
VALIDATION_JSONS = "/home/nick/Workspace/test/validation_results"

scoreable_tasks = {
    "manipulation":scoring_pb2.MANIPULATION,
    "splice": scoring_pb2.SPLICE
}

def get_req(path, task):
    return scoring_pb2.DetectionScoreRequest(
            task = scoreable_tasks[task],
            dataset = scoring_pb2.DatasetPaths(
                reference_dir = VALIDATION_REF_DIR,
                reference_gt = VALIDATION_REF_INDEX,
                index = VALIDATION_INDEX
                ),
            options = scoring_pb2.ScoringOptions(
                farstop = 0.05,
                ci = False,
                ciLevel = 0.9,
                dLevel=0.0
                ),
            results = scoring_pb2.AnalyticOutput(
                # results_dir =path.split(""),
                results_file = path
                ))

def test():
    # channel = grpc.insecure_channel('localhost:50051')v
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = scoring_pb2_grpc.ScoringStub(channel)


        req = scoring_pb2.DetectionScoreRequest(
                task = scoring_pb2.MANIPULATION,
                dataset = scoring_pb2.DatasetPaths(
                    reference_dir = VALIDATION_REF_DIR,
                    reference_gt = VALIDATION_REF_INDEX,
                    index = VALIDATION_INDEX
                    ),
                options = scoring_pb2.ScoringOptions(
                    farstop = 0.05,
                    ci = False,
                    ciLevel = 0.9,
                    dLevel=0.0
                    ),
                results = scoring_pb2.AnalyticOutput(
                    results_dir = "data/test_suite/detectionScorerTests/baseline",
                    results_file = "Base_NC2017_Manipulation_ImgOnly_p-copymove_01.csv"
                    ))
        # resp = scoring_pb2.DetectionScore()
        print("initiating protos")
        print("Request")
        print(str(req))
        print("")
        print("Sending request to server")
        resp = stub.ScoreDetection(req)
        print("Received response")
        print(str(resp))
        print("Exiting")

def score_validation(json_path, write_path = "/tmp/validation_csv_files/",make_csv=True):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = scoring_pb2_grpc.ScoringStub(channel)
        filename = json_path.split("/")[-1]
        outfile, ext = os.path.splitext(filename)
        csv_filename = outfile+".csv"
        # csv_path =  os.path.join(write_path, outfile+".csv")
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        results, masks, task = parse_json(json_path)

        id = str(uuid.uuid4())
        csv_path=id + ".csv"

        if not write_nist_csv(results, csv_path):
            return None
        else:
            req = get_req(csv_path, task)
            # return req
            # return
            return stub.ScoreDetection(req)

if __name__ == "__main__":
    from pprint import pprint
    json = "/home/nick/Workspace/test/validation_results/dmc-take-home-jsons/fibber-fourigh-proto.json"
    resp = score_validation(json)
    pprint(resp)
