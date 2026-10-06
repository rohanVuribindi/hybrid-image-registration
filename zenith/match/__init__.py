"""
ZENITH Feature Matching Module.
Exposes common MatcherResult, BaseMatcher, RootSIFT, LightGlue, RoMa, LoFTR, XoFTR,
and the Multi-Matcher Consensus Ensemble & Adaptive Router.
"""
from .common import MatcherResult, BaseMatcher
from .rootsift import RootSIFTMatcher, RootSIFTAdapter, extract_rootsift_features
from .loftr_matcher import LoFTRMatcher, LoFTRAdapter
from .lightglue_matcher import LightGlueMatcher
from .roma_matcher import RoMaMatcher
from .xoftr_matcher import XoFTRMatcher
from .ensemble import EnsembleMatcher
from .router import AdaptiveMatcherRouter, analyze_image_pair_characteristics
from .hybrid import HybridMatcher

__all__ = [
    "MatcherResult",
    "BaseMatcher",
    "RootSIFTMatcher",
    "RootSIFTAdapter",
    "extract_rootsift_features",
    "LoFTRMatcher",
    "LoFTRAdapter",
    "LightGlueMatcher",
    "RoMaMatcher",
    "XoFTRMatcher",
    "EnsembleMatcher",
    "AdaptiveMatcherRouter",
    "analyze_image_pair_characteristics",
    "HybridMatcher"
]
