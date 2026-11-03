"""MSR-3D: fixed-weight inference for open-vocabulary instance matching."""
from .features import QueryFeatures, load_query
from .model import Matcher

__version__ = '1.1.0'
__all__ = ['Matcher', 'QueryFeatures', 'load_query']
