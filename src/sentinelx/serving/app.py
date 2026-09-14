"""FastAPI application for Sentinel-X model serving (Phase 10).

Route handlers are intentionally thin: validate input (via typed Pydantic
schemas), call a function in :mod:`sentinelx.serving.services`, and return a
typed response. No ML logic lives here — the services layer owns all model
interaction, and the runtime layer owns loading/caching.

Endpoints
---------
    GET  /health
    POST /ingest
    POST /forecast
    GET  /network/state
    GET  /network/history
    GET  /forecast/trajectory
    GET  /risk
    GET  /uncertainty
    GET  /novelty
    GET  /propagation
    GET  /explainability
    POST /counterfactual
    GET  /mitre
    GET  /experiments
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from . import services
from .runtime import SentinelRuntime, get_runtime
from .schemas import (
    CounterfactualRequest, CounterfactualResponse, ExperimentsResponse,
    ExplainabilityResponse, ForecastRequest, ForecastResponse,
    ForecastTrajectory, HealthResponse, IngestRequest, MitreResponse,
    NetworkHistoryResponse, NetworkState, NoveltyResponse, PropagationResponse,
    RiskResponse, StabilityResponse, TrajectoryResponse, UncertaintyResponse,
)


def _runtime_dep() -> SentinelRuntime:
    return get_runtime()


def _handle(fn):
    """Call a service function, mapping ServiceError -> HTTPException."""
    try:
        return fn()
    except services.ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentinel-X Serving API",
        version="0.1.0",
        description=(
            "Serves the trained Sentinel-X network world model and its "
            "analytical layers (forecast, risk, uncertainty, novelty, MITRE "
            "trajectory, propagation, explainability, counterfactual, "
            "stability). Clean service layers; no ML logic in routes; typed "
            "Pydantic schemas; internal tensors are never exposed."),
    )

    # ---------------------------------------------------------------- health #
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(rt: SentinelRuntime = Depends(_runtime_dep)) -> HealthResponse:
        return HealthResponse(**services.health(rt))

    # ---------------------------------------------------------------- ingest #
    @app.post("/ingest", tags=["forecast"])
    def ingest(body: IngestRequest,
               rt: SentinelRuntime = Depends(_runtime_dep)):
        windows = [w.model_dump() for w in body.windows]
        return _handle(lambda: services.ingest(rt, windows, body.horizons))

    # -------------------------------------------------------------- forecast #
    @app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
    def forecast(body: ForecastRequest,
                 rt: SentinelRuntime = Depends(_runtime_dep)) -> ForecastResponse:
        payload = _handle(lambda: services.full_forecast(
            rt, body.t_index, horizons=body.horizons,
            include_explainability=body.include_explainability,
            include_propagation=body.include_propagation,
            include_uncertainty=body.include_uncertainty,
            include_novelty=body.include_novelty,
            include_mitre=body.include_mitre,
            include_stability=body.include_stability,
            mc_passes=body.mc_passes))
        return ForecastResponse(**payload)

    # ------------------------------------------------------------ network/* #
    @app.get("/network/state", response_model=NetworkState, tags=["network"])
    def network_state(t_index: Optional[int] = Query(None),
                      rt: SentinelRuntime = Depends(_runtime_dep)) -> NetworkState:
        return NetworkState(**_handle(lambda: services.network_state(rt, t_index)))

    @app.get("/network/history", response_model=NetworkHistoryResponse,
             tags=["network"])
    def network_history(limit: Optional[int] = Query(None, ge=1),
                        rt: SentinelRuntime = Depends(_runtime_dep)
                        ) -> NetworkHistoryResponse:
        return NetworkHistoryResponse(
            **_handle(lambda: services.network_history(rt, limit)))

    # ---------------------------------------------------- forecast/trajectory #
    @app.get("/forecast/trajectory", response_model=ForecastTrajectory,
             tags=["forecast"])
    def forecast_trajectory(t_index: Optional[int] = Query(None),
                            horizons: Optional[int] = Query(None, ge=1, le=10),
                            rt: SentinelRuntime = Depends(_runtime_dep)
                            ) -> ForecastTrajectory:
        return ForecastTrajectory(
            **_handle(lambda: services.forecast_trajectory(rt, t_index, horizons)))

    # ------------------------------------------------------------------ risk #
    @app.get("/risk", response_model=RiskResponse, tags=["analysis"])
    def risk(t_index: Optional[int] = Query(None),
             horizons: Optional[int] = Query(None, ge=1, le=10),
             rt: SentinelRuntime = Depends(_runtime_dep)) -> RiskResponse:
        return RiskResponse(**_handle(lambda: services.risk(rt, t_index, horizons)))

    # ----------------------------------------------------------- uncertainty #
    @app.get("/uncertainty", response_model=UncertaintyResponse, tags=["analysis"])
    def uncertainty(t_index: Optional[int] = Query(None),
                    mc_passes: int = Query(30, ge=1, le=200),
                    rt: SentinelRuntime = Depends(_runtime_dep)
                    ) -> UncertaintyResponse:
        return UncertaintyResponse(
            **_handle(lambda: services.uncertainty(rt, t_index, mc_passes)))

    # --------------------------------------------------------------- novelty #
    @app.get("/novelty", response_model=NoveltyResponse, tags=["analysis"])
    def novelty(t_index: Optional[int] = Query(None),
                rt: SentinelRuntime = Depends(_runtime_dep)) -> NoveltyResponse:
        return NoveltyResponse(**_handle(lambda: services.novelty(rt, t_index)))

    # ----------------------------------------------------------- propagation #
    @app.get("/propagation", response_model=PropagationResponse, tags=["analysis"])
    def propagation(t_index: Optional[int] = Query(None),
                    rt: SentinelRuntime = Depends(_runtime_dep)
                    ) -> PropagationResponse:
        return PropagationResponse(
            **_handle(lambda: services.propagation(rt, t_index)))

    # -------------------------------------------------------- explainability #
    @app.get("/explainability", response_model=ExplainabilityResponse,
             tags=["analysis"])
    def explainability(t_index: Optional[int] = Query(None),
                       target: str = Query("state", pattern="^(state|risk)$"),
                       rt: SentinelRuntime = Depends(_runtime_dep)
                       ) -> ExplainabilityResponse:
        return ExplainabilityResponse(
            **_handle(lambda: services.explainability(rt, t_index, target)))

    # -------------------------------------------------------- counterfactual #
    @app.post("/counterfactual", response_model=CounterfactualResponse,
              tags=["analysis"])
    def counterfactual(body: CounterfactualRequest,
                       rt: SentinelRuntime = Depends(_runtime_dep)
                       ) -> CounterfactualResponse:
        return CounterfactualResponse(**_handle(lambda: services.counterfactual(
            rt, body.intervention, body.t_index, body.horizons,
            node_index=body.node_index, src=body.src, dst=body.dst,
            path=body.path, timestep=body.timestep)))

    # ----------------------------------------------------------- mitre / traj #
    @app.get("/mitre", response_model=MitreResponse, tags=["analysis"])
    def mitre(t_index: Optional[int] = Query(None),
              horizons: Optional[int] = Query(None, ge=1, le=10),
              rt: SentinelRuntime = Depends(_runtime_dep)) -> MitreResponse:
        return MitreResponse(**_handle(lambda: services.mitre(rt, t_index, horizons)))

    @app.get("/trajectory", response_model=TrajectoryResponse, tags=["analysis"])
    def trajectory(t_index: Optional[int] = Query(None),
                   horizons: Optional[int] = Query(None, ge=1, le=10),
                   rt: SentinelRuntime = Depends(_runtime_dep)
                   ) -> TrajectoryResponse:
        return TrajectoryResponse(
            **_handle(lambda: services.attack_trajectory(rt, t_index, horizons)))

    # ------------------------------------------------------------- stability #
    @app.get("/stability", response_model=StabilityResponse, tags=["analysis"])
    def stability(t_index: Optional[int] = Query(None),
                  horizons: Optional[int] = Query(None, ge=1, le=10),
                  rt: SentinelRuntime = Depends(_runtime_dep)) -> StabilityResponse:
        return StabilityResponse(
            **_handle(lambda: services.stability(rt, t_index, horizons)))

    # ----------------------------------------------------------- experiments #
    @app.get("/experiments", response_model=ExperimentsResponse, tags=["research"])
    def experiments(rt: SentinelRuntime = Depends(_runtime_dep)
                    ) -> ExperimentsResponse:
        return ExperimentsResponse(**_handle(lambda: services.experiments(rt)))

    # --------------------------------------------------------------- startup #
    @app.on_event("startup")
    def _warmup() -> None:
        """Prime the runtime so the FIRST client request is fast, not cold.

        Without this, the first /forecast pays the checkpoint load + sample
        build + OOD fit (~tens of seconds on CPU); a client timeout then aborts
        it and retries into the still-initialising server, which looks like the
        app hanging on the loading screen. Best-effort; never blocks liveness of
        /health beyond the one-time cost.
        """
        try:
            services.warmup(get_runtime())
        except Exception:  # pragma: no cover - defensive
            pass

    return app


# Module-level app for `uvicorn sentinelx.serving.app:app`.
app = create_app()
