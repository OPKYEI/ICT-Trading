# src/utils/monkey_patch_pdarrays.py

import pandas as pd
from features.pd_arrays import PDArrays

# Save a reference to the original method
_original_identify = PDArrays.identify_breaker_blocks

def patched_identify_breaker_blocks(self, df, order_blocks, swing_points):
    """
    For any order_block whose start_idx is a Timestamp, convert it
    to an integer location before calling the original method.
    """
    for ob in order_blocks:
        if isinstance(ob.start_idx, pd.Timestamp):
            ob.start_idx = df.index.get_loc(ob.start_idx)
    # Call the original logic (now that start_idx are ints)
    return _original_identify(self, df, order_blocks, swing_points)

# Overwrite the class method
PDArrays.identify_breaker_blocks = patched_identify_breaker_blocks
