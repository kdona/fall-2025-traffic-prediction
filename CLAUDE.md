# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Arizona traffic prediction project that combines multiple data sources (AZ511 work zones, INRIX traffic data, WZDx events) to analyze and predict traffic patterns on Arizona highways, with a focus on I-10 and Broadway corridor.

## Environment Setup

**Conda Environment** (Recommended):
```bash
conda env create -f environment.yml
conda activate kafka
```

**Pip Installation** (Alternative):
```bash
pip install -r requirements.txt
```

**Environment Variables**:
Copy `.env.template` to `.env` and add required API keys:
- `AZ511_API_KEY`: For AZ511 work zone data
- `INRIX_APP_ID` and `INRIX_APP_KEY`: For INRIX traffic data
- `KAFKA_BOOTSTRAP_SERVERS`: For Kafka streaming (if using)
- PostgreSQL credentials (if using database backend)

## Data Collection Workflow

### 1. Fetch AZ511 Work Zone Events
```bash
python database/az511.py
```
- Fetches work zone events from AZ511 API
- Stores in `database/az511.db` SQLite database
- Schema includes: ID, RoadwayName, EventType, EventSubType, coordinates, severity, timestamps
- Handles datetime conversions (epoch to datetime)
- Primary event types: `roadwork`, `accidentsAndIncidents`, `closures`, `specialEvents`, `restrictionClass`

### 2. Fetch WZDx Work Zone Data
```bash
python database/wzdx.py
```
- Fetches WZDx-format work zone data from AZ511 API
- Stores in `database/workzones.db` with three tables:
  - `work_zones`: Main event data
  - `accidents`: Accident-specific events
  - `daily_counts`: Cached daily event counts
- Uses "lazy update" approach for daily counts (updates only when dashboard requests data)
- Runs periodic fetches (default: every 5 minutes)

### 3. Fetch TomTom Traffic Data (if using TomTom instead of INRIX)
```bash
python database/tomtom.py
```
- Samples points along Arizona road polylines from `az_interstates.geojson` and `az_sr.geojson`
- Queries TomTom Traffic Flow API
- Stores in normalized schema: `road_segments` (stable) + `traffic_data` (time-varying)
- Implements batch processing and rate limiting

## Data Analysis Workflow

The project follows a numbered notebook workflow in `notebooks/`:

1. **`1_eda_wzdx.ipynb`**: EDA on AZ511/WZDx work zone events
   - Loads data from `database/az511.db`
   - Analyzes event types, temporal patterns, spatial distribution
   - Exports processed data to parquet files by EventType:
     - `database/events_accidentsAndIncidents.parquet`
     - `database/events_closures.parquet`
     - `database/events_roadwork.parquet`
     - `database/events_specialEvents.parquet`
     - `database/events_restrictionClass.parquet`

2. **`2_eda_inrix.ipynb`**: EDA on INRIX traffic speed data
   - Analyzes large CSV files in `database/inrix-traffic-speed/`
   - Files: `I10-and-I17-1year.csv`, `Loop101-1year.csv`, `tempe-1yr.csv`
   - TMC (Traffic Message Channel) based road segment analysis

3. **`3_eda_i10.ipynb`**: Focused EDA on I-10 corridor

4. **`4_i10_training_data.ipynb`**: Prepares training data for I-10/Broadway
   - Joins INRIX traffic data with AZ511 events
   - Creates time-binned features (hourly aggregation)
   - Outputs to `database/i10-broadway/`:
     - `X_tensor_1h.npz`: 3D tensor (TMC × time × features)
     - `X_full_1h.parquet`: Full feature dataframe with MultiIndex (tmc_code, time_bin)
   - Features: speed, travel_time_seconds, evt_duration, evt_cat_* (5 categories), miles

5. **`5_i10_train.ipynb`**: Trains prediction models
   - Loads prepared training data from `database/i10-broadway/`
   - Uses balanced sampling (events vs. no-events)
   - Tests multiple models: LinearRegression, Ridge, interactions
   - Target variable: `travel_time_seconds`
   - Evaluation: RMSE, R², cross-validation

## Dashboard Applications

### Main Transportation Dashboard
```bash
streamlit run dashboard/az511app.py
```
- Combined AZ511 events + TomTom traffic visualization
- Interactive map with Plotly
- City-based filtering, date selection, FRC (Functional Road Class) filtering
- Analytics charts: event distributions, duration analysis

### WZDx-Specific Dashboard
```bash
streamlit run dashboard/wzdxapp.py
```
- Focused on WZDx work zone data
- Accident tracking and visualization

### INRIX Dashboard
```bash
streamlit run dashboard/inrixapp.py
```
- INRIX traffic speed visualization

## Key Architecture Patterns

### Data Flow
1. **Collection**: API scripts → SQLite databases
2. **Processing**: Notebooks load from databases → Clean/transform → Export parquet files
3. **Training Data**: Join traffic + events → Time-binned features → Tensor/DataFrame
4. **Modeling**: Load prepared data → Train/evaluate models
5. **Visualization**: Dashboards read from databases + parquet files

### Database Schema Conventions

**AZ511 Database** (`database/az511.db`):
- Single `events` table with 30 columns
- Timestamps stored as Unix epoch integers
- Convert to datetime in notebooks: `pd.to_datetime(df['Reported'], unit='s')`

**WZDx Database** (`database/workzones.db`):
- `work_zones`: Core work zone data with GeoJSON in `raw_data` column
- `accidents`: Subset filtered by "AccidentIncident" in description
- `daily_counts`: Cached aggregations with `last_updated` timestamp

**Training Data Structure**:
- MultiIndex DataFrames: `(tmc_code, time_bin)` as index
- 3D tensor format: `(n_tmc, n_timesteps, n_features)`
- Event categories one-hot encoded: `evt_cat_major`, `evt_cat_minor`, `evt_cat_closure`, `evt_cat_obstruction`, `evt_cat_misc`

### Temporal Handling
- All notebooks use Arizona timezone: `ZoneInfo("America/Phoenix")` (MST, no DST)
- Time binning typically hourly: `resample('1H')`
- Event duration calculated: `(PlannedEndDate - StartDate).dt.total_seconds() / 3600`

### Geospatial Data
- GeoJSON files in `database/`: `az_interstates.geojson`, `az_sr.geojson`, `phx_polygon.geojson`
- Coordinates: longitude (x) comes first in GeoJSON, latitude (y) second
- City filtering uses ±0.1 degree radius (~11km)

## Common Development Tasks

**Adding a New Feature to Training Data**:
1. Modify `4_i10_training_data.ipynb` to compute the feature
2. Add to `feature_cols` list
3. Re-export `X_tensor_*.npz` and `X_full_*.parquet`
4. Update `5_i10_train.ipynb` to use new feature

**Processing New Event Data**:
1. Run collection script: `python database/az511.py` or `wzdx.py`
2. Re-run `1_eda_wzdx.ipynb` to export updated parquet files
3. Re-run `4_i10_training_data.ipynb` to join with traffic data
4. Re-train models in `5_i10_train.ipynb`

**Testing Dashboard Changes**:
```bash
streamlit run dashboard/az511app.py
# Dashboard auto-reloads on file changes
```

## Testing

The project uses `pytest` for testing:
```bash
pytest  # Run all tests
pytest -v  # Verbose output
```

## Data Files to Note

**Large CSV Files** (gitignored):
- `database/inrix-traffic-speed/*.csv`: Multi-GB INRIX traffic data
- Files are read in chunks or with specific date filters in notebooks

**Parquet Exports** (committed to git):
- `database/events_*.parquet`: Processed event data by type
- `database/i10-broadway/*.parquet`: Prepared training data
- Fast read/write, compressed format

**Database Files** (gitignored):
- `database/az511.db`, `database/wzdx.db`, `database/tomtom.db`
- SQLite format, recreated by running collection scripts

## Important Conventions

1. **Notebook Numbering**: Follow the sequence 1-5 for reproducible workflow
2. **Database Paths**: Use `Path(__file__).parent / 'filename.db'` for scripts to work from any directory
3. **Event Categorization**: Map EventSubType to 5 categories (major, minor, closure, obstruction, misc)
4. **TMC Codes**: INRIX uses TMC codes to identify road segments; preserve as index
5. **Time Bins**: Default to hourly aggregation for consistency across datasets
