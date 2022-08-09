#!/usr/bin/env python3

import unittest
from abc import abstractmethod
from typing import List, Callable, Tuple, Union, Optional, Any
from types import NoneType
from math import inf
import time
import subprocess
import os
from test_integration_base import TestIntegrationBase, run_with_args


class TestIntegrationStandaloneC(unittest.TestCase):

    @classmethod
    def compile(cls):
        
    pass