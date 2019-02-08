from __future__ import print_function, division, unicode_literals, absolute_import

import contextlib
import json
import logging
import os
import select
import sys
import threading
import time

from concurrent import futures

import scoring_pb2
import scoring_pb2_grpc

import grpc
# from grpc_health.v1 import health
# from grpc_health.v1 import health_pb2
# from grpc_health.v1 import health_pb2_grpc

from google.protobuf import json_format
logging.basicConfig(level=logging.INFO)



class _ScoringServicer(scoring_pb2_grpc.ScoringServicer):
    """Class registered with gRPC"""

    def __init__(self, svc):
        """Create a servicer using the given Service object as an implementation"""
        self.svc = svc

    def ScoreDetection(self,req,ctx):
        logging.info("Detection scoring request received")
        return self.svc._CallEndpoint(self.svc.DETECTION, req, scoring_pb2.DetectionScore(), ctx)


class ScoringService:
    """

    """

    DETECTION='DetectionScore'
    MASK="MaskScore"

    _ALLOWED_IMPLS= frozenset([DETECTION, MASK])

    def __init__(self, scorer_port=50051, max_workers=10):
        """Implementation of the scoring service"""
        self._impls={}
        self.scorer_port = scorer_port
        self.max_workers = max_workers

    def Start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers),
                             options=(('grpc.sp_reuseport',0),))
        scoring_pb2_grpc.add_ScoringServicer_to_server(_ScoringServicer(self), server)
        if not server.add_insecure_port('[::]:{:d}'.format(self.scorer_port)):
            raise RuntimeError("can't bind to port {!s}: already in use".format(self.scorer_port))
        server.start()
        logging.info("Scoring server started on port {} with PID {}".format(self.scorer_port, os.getpid()))
        return server

    def Run(self):
        server = self.Start()

        try:
            while True:
                time.sleep(3600 * 24)
        except KeyboardInterrupt:
            server.stop(0)
            logging.info("Server stopped")
            return 0
        except Exception as e:
            server.stop(0)
            logging.error("Caught exception: {!s}".format(e))
            return -1

    def RegisterDetection(self, f):
        return self._RegisterImpl(self.DETECTION, f)

    def RegisterSplice(self, f):
        raise NotImplementedError

    def _RegisterImpl(self, type_name, f):
        if type_name not in self._ALLOWED_IMPLS:
            raise ValueError("unkown implementation type {!s} specified".format(type_name))
        if type_name in self._impls:
            raise ValueError("implementation for {!s} already present".format(type_name))
        self._impls[type_name] = f

        return self

    def _CallEndpoint(self, ep_type, req, resp, ctx):
        """
        """

        ep_func = self._impls.get(ep_type)
        logging.info("EPFUNC: {!s}".format(ep_func))
        if not ep_func:
            ctx.abort(grpc.StatusCode.UNIMPLEMENTED, "Endpoint {!r} not implemented".format(ep_type))

        try:
            logging.info("Calling function for {!s}".format(ep_type))
            ep_func(req, resp)
        except ValueError as e:
            logging.exception("invalid input")
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "Endpoint {!s} invalid input: {!s}".format(ep_type, e))
        except NotImplementedError as e:
            logging.warn("Got exception {!s}.  Unimplemented endpoint {!s}".format(e, ep_type))
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "Endpoint {!s} invalid input: {!s}".format(ep_type, e))
        except Exception as e:
            logging.exception("Unknown error: {!s}".format(e))
            ctx.abort(grpc.StatusCode.UNKNOWN, "Error processing endpoint: {!s}.  Got {!s}".format(ep_type, e))
        return resp
