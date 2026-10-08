"""Module 3: SQLite Ledger Persistence (D3) and Enriched Parquet Exporter (D4).

Enforces:
- D3 Contract: SQLite database (anomalies.db) with 'observations' and 'canonical_anomalies' tables.
- D4 Contract:
    * enriched_edges.parquet (width_dir_m, lanes_effective, length_m, geometry).
    * anomalies.parquet (confirmed anomalies with s, t, lon, lat, sigma_s, sigma_t).
"""

from __future__ import annotations
import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from irn.common.logger import log_event
from irn.m3_fusion.deduplication import CanonicalAnomaly


class FusionStorage:
    """Manages SQLite storage (D3) and GeoParquet export (D4)."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite_tables()

    def _init_sqlite_tables(self) -> None:
        """Initialize SQLite ledger schema."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id TEXT,
                    frame_idx INTEGER,
                    frame_ts REAL,
                    edge_id TEXT,
                    class_name TEXT,
                    conf REAL,
                    s_obs REAL,
                    t_obs REAL,
                    sigma_s REAL,
                    sigma_t REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS canonical_anomalies (
                    anomaly_id TEXT PRIMARY KEY,
                    edge_id TEXT,
                    class_name TEXT,
                    s REAL,
                    t REAL,
                    var_s REAL,
                    var_t REAL,
                    confidence REAL,
                    observation_count INTEGER,
                    state TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def persist_observations(self, obs_list: List[Dict[str, Any]]) -> int:
        """Insert batch of raw observations into append-only SQLite ledger."""
        if not obs_list:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany("""
                INSERT INTO observations (
                    sequence_id, frame_idx, frame_ts, edge_id, class_name,
                    conf, s_obs, t_obs, sigma_s, sigma_t
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    o.get("sequence_id", ""),
                    int(o.get("frame_idx", 0)),
                    float(o.get("frame_ts", 0.0)),
                    str(o.get("edge_id", "")),
                    str(o.get("class_name", "")),
                    float(o.get("conf", 0.0)),
                    float(o.get("s_obs", 0.0)),
                    float(o.get("t_obs", 0.0)),
                    float(o.get("sigma_s", 1.0)),
                    float(o.get("sigma_t", 1.0))
                )
                for o in obs_list
            ])
            conn.commit()
        return len(obs_list)

    def persist_canonical_anomalies(self, anomalies: List[CanonicalAnomaly]) -> int:
        """Upsert canonical anomalies into SQLite table."""
        if not anomalies:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany("""
                INSERT INTO canonical_anomalies (
                    anomaly_id, edge_id, class_name, s, t, var_s, var_t,
                    confidence, observation_count, state, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(anomaly_id) DO UPDATE SET
                    s = excluded.s,
                    t = excluded.t,
                    var_s = excluded.var_s,
                    var_t = excluded.var_t,
                    confidence = excluded.confidence,
                    observation_count = excluded.observation_count,
                    state = excluded.state,
                    updated_at = CURRENT_TIMESTAMP
            """, [
                (
                    a.anomaly_id,
                    a.edge_id,
                    a.class_name,
                    a.s,
                    a.t,
                    a.var_s,
                    a.var_t,
                    a.confidence,
                    a.observation_count,
                    a.state
                )
                for a in anomalies
            ])
            conn.commit()
        return len(anomalies)

    def load_canonical_anomalies(self) -> List[CanonicalAnomaly]:
        """Load existing canonical anomalies from SQLite ledger for multi-pass fusion."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT anomaly_id, edge_id, class_name, s, t, var_s, var_t,
                       confidence, observation_count, state
                FROM canonical_anomalies
            """)
            rows = cur.fetchall()

        results: List[CanonicalAnomaly] = []
        for r in rows:
            results.append(CanonicalAnomaly(
                anomaly_id=r[0],
                edge_id=r[1],
                class_name=r[2],
                s=float(r[3]),
                t=float(r[4]),
                var_s=float(r[5]),
                var_t=float(r[6]),
                confidence=float(r[7]),
                observation_count=int(r[8]),
                state=str(r[9]),
            ))
        return results


    def export_d4_parquet(
        self,
        enriched_edges_df: pd.DataFrame,
        canonical_anomalies: List[CanonicalAnomaly],
        out_dir: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """Export Contract D4: enriched_edges.parquet and anomalies.parquet."""
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        # 1. enriched_edges.parquet
        edges_out = out / "enriched_edges.parquet"
        # Convert shapely geometries to WKT string if necessary for standard parquet serialization
        edges_clean = enriched_edges_df.copy()
        if "geometry" in edges_clean.columns:
            edges_clean["geometry_wkt"] = edges_clean["geometry"].apply(lambda g: g.wkt if hasattr(g, "wkt") else str(g))
            edges_clean = edges_clean.drop(columns=["geometry"])

        table_edges = pa.Table.from_pandas(edges_clean)
        pq.write_table(table_edges, edges_out, compression="snappy")
        log_event(f"Exported D4 Enriched Edges to {edges_out}", reason_code="D4-EXPORT-EDGES")

        # 2. anomalies.parquet (Only CONFIRMED anomalies)
        anom_out = out / "anomalies.parquet"
        confirmed = [a for a in canonical_anomalies if a.state == "CONFIRMED"]

        anom_records = [
            {
                "anomaly_id": a.anomaly_id,
                "edge_id": a.edge_id,
                "class_name": a.class_name,
                "s": a.s,
                "t": a.t,
                "sigma_s": a.sigma_s,
                "sigma_t": a.sigma_t,
                "confidence": a.confidence,
                "observation_count": a.observation_count,
                "state": a.state,
            }
            for a in confirmed
        ]

        if not anom_records:
            # Empty schema
            schema = pa.schema([
                ("anomaly_id", pa.string()),
                ("edge_id", pa.string()),
                ("class_name", pa.string()),
                ("s", pa.float64()),
                ("t", pa.float64()),
                ("sigma_s", pa.float64()),
                ("sigma_t", pa.float64()),
                ("confidence", pa.float64()),
                ("observation_count", pa.int64()),
                ("state", pa.string()),
            ])
            table_anom = pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
        else:
            table_anom = pa.Table.from_pandas(pd.DataFrame(anom_records))

        pq.write_table(table_anom, anom_out, compression="snappy")
        log_event(
            f"Exported D4 Anomalies ({len(confirmed)} confirmed) to {anom_out}",
            reason_code="D4-EXPORT-ANOMALIES"
        )

        return edges_out, anom_out
