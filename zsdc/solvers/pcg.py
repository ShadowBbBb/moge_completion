import inspect
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve


def solve_system(A, b, config=None):
    config = config or {}
    start = time.perf_counter()
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    if not np.isfinite(A.data).all() or not np.isfinite(b).all() or np.any(A.diagonal() <= 0):
        raise ValueError("Invalid sparse system")
    rtol = config.get("rtol", 1e-6)
    if rtol <= 0 or config.get("maxiter", 1000) <= 0:
        raise ValueError("CG tolerance and iteration limit must be positive")
    if config.get("solver", "cg") == "direct":
        solution, code = spsolve(A, b), 0
    elif config.get("solver", "cg") == "cg":
        # SciPy 1.8 uses tol; recent SciPy uses rtol.
        tolerance = {"rtol" if "rtol" in inspect.signature(cg).parameters else "tol": rtol}
        solution, code = cg(A, b, M=sparse.diags(1/A.diagonal()), atol=0.,
                            maxiter=config.get("maxiter", 1000), callback=callback, **tolerance)
    else:
        raise ValueError("solver must be cg or direct")
    residual = float(np.linalg.norm(A @ solution-b))
    relative = residual / max(float(np.linalg.norm(b)), 1e-15)
    converged = bool(code == 0 and np.isfinite(solution).all() and np.isfinite(relative) and relative <= max(rtol*10, 1e-10))
    return solution, {"converged": converged, "cg_info": int(code), "iterations": iterations,
                      "residual": residual if np.isfinite(residual) else None,
                      "relative_residual": relative if np.isfinite(relative) else None,
                      "solve_seconds": time.perf_counter()-start, "nnz": int(A.nnz),
                      "system_bytes": int(A.data.nbytes+A.indices.nbytes+A.indptr.nbytes+b.nbytes)}
