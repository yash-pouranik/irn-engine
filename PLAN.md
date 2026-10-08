# Automated Road Network Modeling Engine for Lane-Free Indian Traffic Simulations
## Master Engineering & Phased Implementation Plan (`PLAN.md`)
**Institution:** Shri Vaishnav Institute of Information Technology (SVIIT), Indore  
**Program:** B.Tech CSE Major Project (2026–27)  
**Document Version:** 1.0 (Aligned with SRS v1.0)  

---

## 1. Executive Summary & Project Mission

Autonomous driving and microscopic traffic simulators (SUMO, CARLA) inherently assume lane-disciplined, homogeneous traffic operating on well-maintained road networks. Indian urban roads fundamentally violate these assumptions: traffic is highly heterogeneous (two-wheelers, auto-rickshaws, cars, heavy vehicles, pedestrians) and effectively **lane-free**, while roads feature localized, unmapped anomalies (potholes, speed breakers, barricades, illegal parking).

This project implements **IRN (`irn`)**, a modular, batch-oriented Python data engineering pipeline that:
1. Ingests static road network topology from **OpenStreetMap (OSM)** via Overpass/OSMnx.
2. Ingests localized road anomaly detections and vehicle observations from the **IIIT-Hyderabad Indian Driving Dataset (IDD Multimodal)**.
3. Performs mathematically rigorous geospatial data fusion: camera ground-plane ray projection, Hidden Markov Model (HMM) map matching, relative linear referencing $(s, t)$, and multi-pass Mahalanobis anomaly de-duplication.
4. Compiles the enriched network into **ASAM OpenDRIVE (`.xodr`)** and **Eclipse SUMO (`.net.xml`, `.sumocfg`)** natively configured with the **Sublane Model** for realistic, lane-free Indian traffic simulation.

```
+--------------------+       +----------------------+
| OpenStreetMap(OSM) |       | IDD Multimodal (IIIT)|
+---------+----------+       +----------+-----------+
          |                             |
          v [M1: Yashika]               v [M2: Vipul]
  D1: Road Graph GPKG           D2: detections.parquet
          |                             |
          +--------------+--------------+
                         |
                         v [M3: Yash]
            Geospatial Data Fusion Engine
             - Ground-Plane Ray Projection
             - HMM Viterbi Map Matching
             - Relative Linear Referencing (s, t)
             - 2-Level Mahalanobis De-duplication
             - Road Width Fusion (OSM + Satellite)
                         |
                         v
            D3: Anomaly Store (SQLite)
            D4: Enriched GeoParquet
                         |
                         v [M4: Vivek]
            Simulation Compiler
             - SUMO Plain-XML -> netconvert
             - Indian Vehicle Profiles (vtypes.add.xml)
             - Sublane Model (scenario.sumocfg)
             - ASAM OpenDRIVE 1.6.1 (network.xodr)
                         |
                         v
      +-------------------------------------+
      | Zero-Error Ready-to-Run Simulation  |
      |   (SUMO-GUI & CARLA Visualizations) |
      +-------------------------------------+
```

---

## 2. Team Ownership & Responsibility Matrix

| Module | Team Lead | Ownership Scope | Core Libraries / Tools |
| :--- | :--- | :--- | :--- |
| **M1: Map Fetcher** | **Yashika Goswami** | OSM acquisition via bounding box/place, caching, attribute parsing (lanes, speed, width), UTM projection, graph cleanup | `osmnx >= 2.0`, `geopandas`, `shapely`, `pyproj` |
| **M2: CV Layer** | **Vipul Yadav** | IDD Multimodal video inference, YOLOv8 object detector for Indian road anomalies & agents, ByteTrack tracking, Parquet export | `ultralytics`, `opencv-python`, `pyarrow`, Google Colab T4 GPU |
| **M3: Geospatial Fusion** *(Core)* | **Yash Pouranik** | Temporal sync (GPS $\leftrightarrow$ video), camera ray projection with covariance, cKDTree + HMM Viterbi map matching, linear referencing, multi-pass de-duplication, SQLite store | `numpy`, `scipy`, `shapely >= 2.0`, `pyproj`, `sqlite3` |
| **M4: Simulation Compiler** | **Vivek Tekwani** | Plain-XML builder, `netconvert` left-hand runner, Indian vehicle types with arbitrary lateral alignment, speed zone generation, OpenDRIVE `.xodr` builder | `lxml`, `sumo-tools`, Eclipse SUMO `>= 1.18` |

---

## 3. Data Contracts & File Interfaces

The entire pipeline communicates strictly via versioned, validated files (Constraint C-4).

### 3.1 D1: Road Graph (`data/d1_graph/`)
- `edges.gpkg`: `edge_id` (u-v-key), `u`, `v`, `highway`, `oneway`, `lanes`, `width_osm_m`, `maxspeed_kph`, `surface`, `length_m`, LineString geometry (WGS84 & projected UTM).
- `nodes.gpkg`: `node_id`, `lon`, `lat`, `highway` (traffic signals, crossings).
- `osm_snapshot.json`: Area query, snapshot timestamp, OSMnx version, attribution string.

### 3.2 D2: Detections Schema (`data/d2_detections/detections.parquet`)
- `sequence_id` (str), `frame_idx` (int), `frame_ts` (float seconds).
- `class_name` (str): Static (`pothole`, `speed_breaker`, `barrier`, `barricade`, `parked_vehicle_obstruction`) / Dynamic (`car`, `two_wheeler`, `auto_rickshaw`, `bus`, `truck`, `bicycle`, `cycle_rickshaw`, `pedestrian`, `stray_animal`).
- `conf` (float $\ge 0.25$), `x1`, `y1`, `x2`, `y2`, `img_w`, `img_h` (ground contact pixel is bottom-center: $\frac{x_1 + x_2}{2}, y_2$).
- `track_id` (optional int from ByteTrack), `model_version` (str).

### 3.3 D3: Persistent Anomaly Store (`data/d3_store/anomalies.db`)
- SQLite database with tables: `observations` (append-only ledger), `canonical_anomalies` (UUID, state, coordinates), `agent_stats` (class mix per edge), `edge_widths`.

### 3.4 D4: Enriched Network (`data/d4_enriched/`)
- `enriched_edges.parquet`: Edges with fused `width_dir_m`, `lanes_effective`, `class_mix`.
- `anomalies.parquet`: Only `CONFIRMED` anomalies with $(s, t)$ offsets, $(\text{lon}, \text{lat})$, $\sigma_s, \sigma_t$.

### 3.5 Output Scenario Bundle (`output_scenario/`)
- `network.net.xml`: SUMO network compiled in left-hand mode (`--lefthand=true`).
- `network.xodr`: ASAM OpenDRIVE 1.6.1 specification-compliant XML.
- `vtypes.add.xml`: Heterogeneous Indian vehicle definition file.
- `anomalies.add.xml`: Speed-limit drop zones and localized obstacles.
- `scenario.sumocfg`: Configuration launching SUMO with Sublane lateral resolution $0.6\,\text{m}$.
- `preview.geojson`: GeoJSON layer viewable instantly in QGIS / geojson.io.
- `MANIFEST.json` & `ATTRIBUTION.md`: Execution provenance, run statistics, and ODbL license.

---

## 4. Phased Implementation Roadmap

```
+------------------------------------------------------------------------+
| Phase 0: Setup, Schemas & Synthetic Ground Truth Generator (Week 1)   |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| Phase 1: M1 Map Fetcher + M2 Mock Detector & IDD Extraction (Week 2)   |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| Phase 2: M3 Core Part A - Ray Projection & GPS Sync (Week 3)           |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| Phase 3: M3 Core Part B - HMM Map Matching & De-duplication (Week 4)   |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| Phase 4: M4 Simulation Compiler & SUMO Sublane Config (Week 5)         |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| Phase 5: End-to-End Pipeline, Field Validation & Demo (Week 6)         |
+------------------------------------------------------------------------+
```

---

### Phase 0: Foundations, Architecture & Synthetic Ground Truth
**Goal:** Unblock all team members simultaneously so nobody waits for another module to start coding.

- [x] **Task 0.1: Project Skeleton & Tooling**
  - Initialize Git repository with standard package structure (`irn/m1_fetcher`, `irn/m2_cv`, `irn/m3_fusion`, `irn/m4_compiler`, `irn/common`).
  - Configure `pyproject.toml` and lock dependencies (`osmnx>=2.0`, `shapely>=2.0`, `pyproj`, `geopandas`, `scipy`, `numpy`, `pyarrow`, `pydantic`, `lxml`, `typer`).
  - Verified package installation and created configuration templates (`configs/rig.yaml`, `configs/run_config.yaml`).
- [x] **Task 0.2: Pydantic & PyArrow Schemas**
  - Implement `irn/m2_cv_interface/schema.py` enforcing strict D2 detection schema validation.
  - Implement reject logging utility for malformed rows (Constraint C-4, NFR-REL-07).
- [x] **Task 0.3: Synthetic World Ground Truth Generator (NFR-QA-01)**
  - Create `tests/fixtures/synthetic_world.py`:
    - Defines a known synthetic road segment of length $100\,\text{m}$, width $7.5\,\text{m}$.
    - Plants 3 ground-truth potholes at exact known $(s, t)$ coordinates: $(25.0, -1.2)$, $(50.0, 0.8)$, $(75.0, -0.5)$.
    - Generates synthetic GPS trace with realistic Gaussian noise.
    - Projects known 3D points back through camera intrinsics $K$ to produce ground-truth pixel bounding boxes $(u, v)$.
  - *Outcome:* An automated verification test that validates M3 and M4 independently of external datasets.

---

### Phase 1: M1 Map Fetcher & M2 IDD Data Prep
**Goal:** Acquire real road geometry for IDD drive sequences and prepare detection inputs.

- [x] **Task 1.1: OSMnx Fetcher (`irn/m1_fetcher/osm_fetcher.py`)** *(Yashika)*
  - Download drivable road graph using `osmnx.graph_from_bbox` with custom Overpass user agent, backoff, and offline fallback generator.
  - Compute area centroid and determine optimal projected CRS (UTM Zone 43N / 44N for Indian coordinates).
  - Project graph from EPSG:4326 to UTM.
- [x] **Task 1.2: Attribute Normalization & Graph Pruning (`irn/m1_fetcher/cleaner.py`)** *(Yashika)*
  - Parse and normalize `lanes`, `maxspeed`, `width` attributes into typed columns.
  - Retain largest connected component; purge zero-length edges.
  - Export `edges.gpkg`, `nodes.gpkg`, and `osm_snapshot.json` (D1 contract).
- [x] **Task 1.3: IDD Multimodal Adapter & YOLOv8 Inference (`irn/m2_cv_interface/`)** *(Vipul)*
  - IDD adapter for front-view camera frames and vehicle GPS log (`gps.csv`).
  - YOLOv8 detector and deterministic `MockDetector` for anomalies (`pothole`, `speed_breaker`, `barricade`) and dynamic vehicles.
  - Export detections to `detections.parquet` conforming to Contract D2.

---

### Phase 2: M3 Core Part A — Camera Ground-Plane Model & GPS Sync
**Goal:** Convert 2D pixel detections into metric ground positions $(X, Y)$ in the vehicle reference frame.

- [x] **Task 2.1: Rig Calibration Configuration (`configs/rig.yaml`)** *(Yash)*
  - Specify camera intrinsic matrix $K$: focal lengths $(f_x, f_y)$, principal point $(c_x, c_y)$.
  - Specify camera mounting height $h$ (default $1.5\,\text{m}$), pitch angle $\theta_{\text{pitch}}$, roll angle $\theta_{\text{roll}}$, and antenna-to-camera translation vector.
- [x] **Task 2.2: Temporal GPS-Video Synchronizer (`irn/m3_fusion/sync.py`)** *(Yash)*
  - Match frame timestamp $t_{\text{frame}}$ to GPS time: $t_{\text{gps}} = t_{\text{start}} + t_{\text{frame}} + \Delta t_{\text{offset}}$.
  - Perform linear interpolation of vehicle position $(\text{lat}, \text{lon})$ between GPS fixes ($\le 0.2\,\text{s}$ interval).
  - Reject frames without valid GPS fixes within $1.0\,\text{s}$ or speeds $> 45\,\text{m/s}$.
- [x] **Task 2.3: Ground-Plane Ray Intersection (`irn/m3_fusion/camera_model.py`)** *(Yash)*
  - Extract bottom-center pixel $(u, v)$ as the contact point on the road surface.
  - Compute normalized camera ray: $d_c = K^{-1} [u, v, 1]^T$.
  - Transform ray to vehicle frame: $d_v = R(\text{pitch}, \text{roll}) \cdot d_c$.
  - Intersect ray with ground plane $z = -h$: $\lambda = \frac{-h}{d_v[2]}$ (valid only if $d_v[2] < 0$).
  - Obtain ground point: $P_{\text{vehicle}} = \lambda \cdot d_v = (X_{\text{forward}}, Y_{\text{left}})$.
- [x] **Task 2.4: Range Gating & Uncertainty Propagation** *(Yash)*
  - Discard detections outside the operational range gate $[4\,\text{m}, 30\,\text{m}]$.
  - Compute $2 \times 2$ covariance matrix $\Sigma_{XY}$ via Jacobian error propagation from pixel error $\sigma_v$ and pitch uncertainty.
  - **Verification Gate:** Pass synthetic round-trip test (`tests/test_projection.py`) with error $< 1\,\text{cm}$ (FR-M3-10).

---

### Phase 3: M3 Core Part B — HMM Map Matching & Multi-Pass De-duplication
**Goal:** Bind vehicle and anomaly positions to road edges and eliminate duplicate observations.

- [x] **Task 3.1: Candidate Road Edge Indexing (`irn/m3_fusion/map_matcher.py`)** *(Yash)*
  - Sample road centerlines at $5\,\text{m}$ intervals in projected UTM coordinates.
  - Build `scipy.spatial.cKDTree` spatial index over edge sample points.
  - Query nearest $K=8$ candidate edges within $35\,\text{m}$ of vehicle GPS coordinates.
- [x] **Task 3.2: Hidden Markov Model (HMM) Viterbi Decoding** *(Yash)*
  - **Emission Probability:** Gaussian on perpendicular distance $d_{\text{perp}}$ to edge centerline ($\sigma_{\text{GPS}} = 5.0\,\text{m}$):
    $$p(z_t | s_i) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{d_{\text{perp}}^2}{2\sigma^2}\right)$$
  - **Transition Probability:** Penalty based on discrepancy between shortest route distance on road network and Euclidean distance:
    $$p(s_j | s_i) = \frac{1}{\beta} \exp\left(-\frac{|\text{dist}_{\text{route}} - \text{dist}_{\text{euclid}}|}{\beta}\right)$$
  - **Heading Filter:** Reject candidate edges whose bearing angle deviates $> 100^\circ$ from vehicle heading.
  - Execute Viterbi dynamic programming to determine global optimal road edge trajectory sequence.
- [x] **Task 3.3: Relative Linear Referencing (`irn/m3_fusion/linear_ref.py`)** *(Yash)*
  - Project matched vehicle point onto edge geometry using `shapely.line_locate_point` to get $(s_{\text{ego}}, t_{\text{ego}})$.
  - Project relative camera offset $(X, Y)$ onto road frame using heading delta $d\psi = \psi_{\text{ego}} - \tau(s_{\text{ego}})$:
    $$ds = X \cos(d\psi) - Y \sin(d\psi)$$
    $$dt = X \sin(d\psi) + Y \cos(d\psi)$$
    $$s_{\text{obs}} = s_{\text{ego}} + ds, \quad t_{\text{obs}} = t_{\text{ego}} + dt$$
  - Map bidirectional passes to undirected canonical segment ID ($s' = L - s, t' = -t$).
- [x] **Task 3.4: Multi-Pass Anomaly De-duplication (`irn/m3_fusion/deduplication.py`)** *(Yash)*
  - **Level 1 (Within-pass):** Group observations sharing `track_id` into a single pass-level estimate.
  - **Level 2 (Across passes):** Formulate assignment as minimum-weight bipartite matching using `scipy.optimize.linear_sum_assignment` (Hungarian Algorithm).
  - Cost metric: Mahalanobis distance $D_M^2 = (\Delta s, \Delta t) \Sigma^{-1} (\Delta s, \Delta t)^T$.
  - Gating threshold: $\chi^2_{2, 0.99} = 9.21$. Pairs with $D_M^2 > 9.21$ or incompatible classes are never merged.
  - Lifecycle state machine: `CANDIDATE` transitions to `CONFIRMED` upon receiving $\ge 2$ independent passes (or $\ge 5$ consistent frames with $\text{conf} \ge 0.5$).
- [x] **Task 3.5: Road Width Fusion (`irn/m3_fusion/width_fusion.py`)** *(Yash)*
  - Ingest OSM tag widths; fallback to highway classification defaults.
  - Fuse available satellite transect measurements via inverse-variance weighting.
  - Compute directional usable width `width_dir_m` and `lanes_effective = max(1, round(w_dir / 3.0))`.
- [x] **Task 3.6: SQLite Persistence & Parquet Export (`irn/m3_fusion/storage.py`)** *(Yash)*
  - Upsert observations and anomalies into `data/d3_store/anomalies.db`.
  - Export `enriched_edges.parquet` and `anomalies.parquet` (D4 contract).

---

### Phase 4: M4 Simulation Compiler (SUMO & OpenDRIVE)
**Goal:** Transform fused geospatial data into ready-to-run simulator artifacts.

- [ ] **Task 4.1: Plain-XML Network Generator (`irn/m4_compiler/sumo_plain_xml.py`)** *(Vivek)*
  - Read D4 enriched network.
  - Generate `plain.nod.xml` (nodes with UTM coordinates, traffic lights for signalized intersections).
  - Generate `plain.edg.xml` (edges with lengths, speeds, shape coordinates, left-hand rules).
  - Implement `lanefree_mode = collapse`: build single wide lane representing full directional road width without artificial lane dividers.
- [ ] **Task 4.2: Automated `netconvert` Compilation (`irn/m4_compiler/netconvert_runner.py`)** *(Vivek)*
  - Invoke `netconvert` subprocess in left-hand mode:
    ```bash
    netconvert --node-files=plain.nod.xml \
               --edge-files=plain.edg.xml \
               --output-file=network.net.xml \
               --lefthand=true \
               --junctions.join=true
    ```
  - Verify compiled network contains 0 errors (Constraint C-2, FR-M4-10).
- [ ] **Task 4.3: Indian Vehicle Definition Generator (`irn/m4_compiler/vtypes_generator.py`)** *(Vivek)*
  - Create `vtypes.add.xml` defining heterogeneous Indian vehicle fleet:
    - `two_wheeler`: length $1.9\,\text{m}$, width $0.7\,\text{m}$, `lateral-alignment="arbitrary"`, `minGapLat="0.2"`.
    - `auto_rickshaw`: length $2.7\,\text{m}$, width $1.4\,\text{m}$, `lateral-alignment="center"`.
    - `car`: length $4.2\,\text{m}$, width $1.7\,\text{m}$, `lateral-alignment="center"`.
    - `bus`: length $10.5\,\text{m}$, width $2.5\,\text{m}$, `lateral-alignment="center"`.
    - `truck`: length $9.0\,\text{m}$, width $2.5\,\text{m}$, `lateral-alignment="center"`.
    - `bicycle`: length $1.8\,\text{m}$, width $0.6\,\text{m}$, `lateral-alignment="arbitrary"`.
- [ ] **Task 4.4: Dynamic Anomaly Realization (`irn/m4_compiler/anomalies_generator.py`)** *(Vivek)*
  - Create `anomalies.add.xml`:
    - Speed-limit drop zones ($15\text{--}20\,\text{km/h}$) around pothole and speed breaker footprints.
    - Stationary obstacle bounding boxes at precise lateral offsets $t$ for barricades and parked vehicle obstructions.
- [ ] **Task 4.5: Sublane Simulator Configuration (`scenario.sumocfg`)** *(Vivek)*
  - Assemble master SUMO configuration activating sublane modeling:
    ```xml
    <configuration>
        <input>
            <net-file value="network.net.xml"/>
            <additional-files value="vtypes.add.xml,anomalies.add.xml"/>
        </input>
        <processing>
            <lateral-resolution value="0.6"/>
            <step-length value="0.1"/>
        </processing>
    </configuration>
    ```
- [ ] **Task 4.6: ASAM OpenDRIVE 1.6.1 Exporter (`irn/m4_compiler/xodr_builder.py`)** *(Vivek)*
  - Construct valid OpenDRIVE XML (`network.xodr`): header with proj4 geoReference, roads, left-hand traffic rule, junctions.
  - Embed confirmed anomalies as OpenDRIVE `<object>` elements with $(s, t)$ offsets and bounding dimensions.
  - Validate output against official ASAM OpenDRIVE XSD schema.

---

### Phase 5: Pipeline Integration, Validation & Final Demo
**Goal:** Deliver unified CLI, pass all SRS acceptance tests, and prepare high-impact visual demonstrations.

- [ ] **Task 5.1: Master CLI Interface (`irn/cli.py`)**
  - Implement unified Typer CLI supporting modular and end-to-end runs:
    ```bash
    irn run --config configs/run_idd_demo.yaml --out output_scenario/
    irn fetch --bbox 78.34,17.42,78.38,17.46 --out data/d1_graph/
    irn fuse --graph data/d1_graph/ --detections data/d2/detections.parquet --gps data/idd/gps.csv
    irn compile --enriched data/d4_enriched/ --out output_scenario/
    irn report --scenario output_scenario/
    ```
  - Generate `preview.geojson` for QGIS visualization.
- [ ] **Task 5.2: Verification & Smoke Testing (FR-M4-10)**
  - Execute automated 1-second SUMO smoke simulation with sample vehicle flow to verify zero collision/geometry crashes.
  - Validate OpenDRIVE schema compliance using `lxml.etree` and ASAM XSD.
- [ ] **Task 5.3: Benchmark Metric Generation (NFR-REL-02)**
  - Calculate and output anomaly de-duplication precision ($\ge 0.95$), recall ($\ge 0.85$), and duplicate rate ($< 2\%$).
  - Measure execution throughput (target $\ge 20,000$ detections/sec in fusion stages).
- [ ] **Task 5.4: Presentation Demo Preparation**
  - Record side-by-side comparison video:
    - *Standard SUMO:* Rigid lanes, vehicles stuck in unrealistic queues.
    - *IRN Sublane SUMO:* Heterogeneous vehicles freely weaving, motorcycles filtering between cars, natural avoidance of potholes.
  - Prepare interactive QGIS map showing verified anomaly clusters.

---

## 5. Verification Gates & Quality Checklist

| Gate ID | Check Item | Pass Criteria | Command / Tool |
| :--- | :--- | :--- | :--- |
| **G-1** | Projection Accuracy | Synthetic 3D points recovered within $\le 1\,\text{cm}$ | `pytest tests/test_projection.py` |
| **G-2** | Map-Matching Precision | $\ge 95\%$ correct edge assignment on test paths | `pytest tests/test_map_matcher.py` |
| **G-3** | De-duplication Stability | Repeated passes merge into single UUID with shrinking covariance | `pytest tests/test_deduplication.py` |
| **G-4** | SUMO Netconvert | Netconvert compiles plain-XML with 0 errors | `netconvert --node-files=... --edge-files=...` |
| **G-5** | OpenDRIVE Validity | `.xodr` passes official ASAM 1.6.1 XSD validation | `python -m irn.m4_compiler.validator --xodr` |
| **G-6** | Smoke Simulation | SUMO executes 10s simulation with sublane active and 0 errors | `sumo -c output_scenario/scenario.sumocfg --end 10` |

---

## 6. Execution Environment & Dependencies

### Python Environment (Standard 8GB/16GB CPU Laptop)
```bash
# Recommended Python version: 3.11 or 3.12
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Core Scientific & Geospatial Dependencies
pip install osmnx>=2.0 geopandas shapely>=2.0 pyproj rasterio scipy numpy pyarrow pydantic lxml typer pytest
```

### Simulation Toolchain (Lightweight C++ Engine)
- **Eclipse SUMO:** Version $\ge 1.18$
  - Windows: Install via MSI installer (`winget install Eclipse.SUMO`) and verify `SUMO_HOME` environment variable.
  - Linux/WSL2: `sudo apt-get install sumo sumo-tools sumo-doc`

---

## 7. Viva & Presentation Strategy (How to Wow Examiners)

1. **Problem Statement Hook:** Begin by showing a standard lane-disciplined SUMO simulation vs. a real video of Indian traffic, proving traditional simulators fail in India.
2. **Mathematical Defense:** Present the HMM Viterbi transition/emission equations and the ground-plane ray-casting formulas. Examiners give maximum credit to mathematical depth over simple API calls.
3. **Live Demonstration:**
   - Execute `irn run` in terminal to demonstrate pipeline speed.
   - Launch `sumo-gui` with `scenario.sumocfg` and click Play: demonstrate two-wheelers filtering through sublanes and swerving around mapped potholes.
   - Open `preview.geojson` in QGIS to show geographical alignment over satellite imagery.
4. **Standardization Claim:** Emphasize compliance with international standards: **ASAM OpenDRIVE 1.6.1** and **ODbL Share-Alike** licensing.
