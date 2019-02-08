import grpc
import scoreservice
import scoring_pb2
import scoring_pb2_grpc


# channel = grpc.insecure_channel('localhost:50051')
with grpc.insecure_channel('localhost:50051') as channel:
    stub = scoring_pb2_grpc.ScoringStub(channel)


    req = scoring_pb2.DetectionScoreRequest(
            task = scoring_pb2.MANIPULATION,
            dataset = scoring_pb2.DatasetPaths(
                reference_dir = "data/test_suite/detectionScorerTests/reference",
                reference_gt = "NC2017-manipulation-ref.csv",
                index = "NC2017-manipulation-index.csv"
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
