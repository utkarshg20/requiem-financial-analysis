from pydantic import BaseModel
from datetime import datetime

class SourceField(BaseModel):
    name: str; dtype: str; role: str

class Source(BaseModel):
    alias: str; provider: str; dataset: str; schema_version: str
    fields: list[SourceField]; filters: str | None = None

class Join(BaseModel):
    left: str; right: str; keys: list[str]; type: str; point_in_time_on: str

class PitPolicy(BaseModel):
    mode: str; lookback_lag_seconds: int; corporate_actions: str; fx_policy: str

class DataBindingPlan(BaseModel):
    binding_id: str; spec_id: str; created_at: str
    entitlements_ok: bool
    sources: list[Source]; joins: list[Join]; pit_policy: PitPolicy

def plan_polygon_equities(spec) -> DataBindingPlan:
    return DataBindingPlan(
        binding_id=f"bind_{spec.spec_id}",
        spec_id=spec.spec_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        entitlements_ok=True,
        sources=[
            Source(
                alias="prices",
                provider="Polygon",
                dataset="v2_aggs",
                schema_version="2025.01",
                fields=[
                    SourceField(name="ticker", dtype="string", role="key"),
                    SourceField(name="date", dtype="date", role="key"),
                    SourceField(name="close", dtype="float", role="feature"),
                    SourceField(name="volume", dtype="float", role="feature"),
                ],
                filters=None
            )
        ],
        joins=[],
        pit_policy=PitPolicy(
            mode="asof_join",
            lookback_lag_seconds=86400,
            corporate_actions="total_return",
            fx_policy="no_fx",
        ),
    )
