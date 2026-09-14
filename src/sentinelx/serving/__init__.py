"""Sentinel-X Phase-10: model serving.

Exposes the trained Sentinel-X world model (Phase 4) and every analytical layer
built on top of it (Phases 5-9) through a typed FastAPI application.

Design discipline (per the Phase-10 spec):
  * NO ML logic lives in the route handlers. Routes only validate input, call a
    service, and serialise a typed Pydantic response.
  * The heavy lifting (loading the checkpoint, rebuilding the leakage-safe
    K-step samples from the Phase-2 cache, running rollouts / uncertainty / OOD
    / trajectory / explainability / propagation / counterfactual / stability) is
    done in :mod:`sentinelx.serving.services`, which is a thin, well-typed
    adapter over the existing worldmodel + research packages. Nothing is
    retrained and no cache is rebuilt.
  * Internal tensors are never exposed. Latent vectors are returned as plain
    ``list[float]`` only where the frontend needs them (forecast trajectory),
    and raw ``torch.Tensor`` objects never cross the API boundary.

The FastAPI app is created by :func:`sentinelx.serving.app.create_app`.
"""

from .app import create_app

__all__ = ["create_app"]
