"""
StratQ: Simultaneous Per-Group Accuracy Guarantees for Proxy-Accelerated Analytical Queries.
"""

from .bounds import (
    WoRControlVariateCS,
    FixedSampleEB,
    empirical_bernstein_radius,
    hoeffding_radius,
)
from .engine import (
    StratQEngine,
    GroupAggResult,
    GroupSelectResult,
    GroupHavingResult,
    GroupTopKResult,
)

__all__ = [
    "WoRControlVariateCS",
    "FixedSampleEB",
    "empirical_bernstein_radius",
    "hoeffding_radius",
    "StratQEngine",
    "GroupAggResult",
    "GroupSelectResult",
    "GroupHavingResult",
    "GroupTopKResult",
]
