"""Cached public solution with Global and full table visual forward in TensorRT."""
from participant.solution_public_trt_global import Solution as GlobalSolution


class Solution(GlobalSolution):
    trt_stages=('global','table')

