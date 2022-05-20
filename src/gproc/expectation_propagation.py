"""Provides functions for performing Expectation Propagation approximation to GP classification posteriors."""

from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
import numpy as np