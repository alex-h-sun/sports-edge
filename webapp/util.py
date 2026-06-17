"""Small helpers shared by the API routers."""

import math

import numpy as np
from starlette.responses import JSONResponse


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays and non-finite floats to JSON-safe
    Python values. The edge dicts coming out of the model layer can carry np.float64
    and NaN, neither of which strict JSON parsers accept."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def json_response(data, status_code: int = 200) -> JSONResponse:
    return JSONResponse(to_jsonable(data), status_code=status_code)
