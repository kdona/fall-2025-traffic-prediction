# Arizona Transportation Dashboard

A comprehensive transportation analytics dashboard that visualizes Arizona work zone data and real-time traffic flow using AZ511 and TomTom Traffic API integration.

![Work Zone Dashboard](images/workzone.png)
*Interactive map showing AZ511 work zones and traffic data across Arizona*

## Features

- **AZ511 Work Zone Monitoring**
  - Real-time work zone events
  - Construction and incident tracking
  - Geographic distribution analysis
  - Duration and timing analytics

- **TomTom Traffic Flow Integration**
  - Live traffic speed data
  - Color-coded traffic flow visualization
  - Road segment analysis
  - Speed ratio calculations (current vs free-flow)

![TomTom Traffic Flow](images/tomtom.png)
*5-tier color-coded traffic flow visualization with road type filtering*

- **Interactive Visualizations**
  - Combined map view with work zones and traffic flow
  - City-based filtering (Phoenix, Tucson, Flagstaff, Gilbert, Yuma)
  - Date range selection
  - Event type distribution charts
  - Duration analysis histograms

## Prerequisites

- Python 3.10+
- Streamlit
- TomTom API credentials
- SQLite (for data storage)
- Plotly for visualizations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd wzdx
```

2. Install required dependencies:
```bash
pip install streamlit plotly pandas sqlite3 requests python-dotenv numpy
```

3. Set up your TomTom API credentials:
```bash
cp .env.template .env
# Edit .env and add your TOMTOM_API_KEY
```

## Usage

### Data Collection

1. **Collect TomTom Traffic Flow Data**:
```bash
python database/tomtom.py
```
This script:
- Fetches traffic flow data from multiple points across Phoenix area
- Uses batch processing (every 100 API calls) for efficiency
- Stores data in `database/tomtom.db` SQLite database
- Creates unique hashes to avoid duplicate segments

2. **Collect AZ511 Work Zone Data**:
```bash
python database/az511.py
```
This script:
- Fetches work zone events from AZ511 API
- Stores data in `database/az511.db` SQLite database
- Includes construction, incidents, and road closures

### Dashboard

**Start the Transportation Dashboard**:
```bash
streamlit run dashboard/az511app.py
```

Dashboard Features:
- **Data Source Selection**: Toggle between AZ511 work zones and TomTom traffic flow
- **Geographic Filtering**: Filter by city (Phoenix, Tucson, Flagstaff, Gilbert, Yuma)
- **Date Selection**: View data for specific dates
- **Interactive Map**: 
  - AZ511 events shown as colored markers (by event type)
  - TomTom traffic flow shown as colored lines with 5-tier system:
    - Green: Excellent flow (90%+ free flow speed)
    - Light Green: Good flow (70-90%)
    - Yellow: Moderate flow (50-70%)
    - Orange: Slow flow (30-50%)
    - Red: Very slow/stopped (<30%)
  - Road type filtering (FRC0-FRC6: Motorways to Local roads)
  - Performance optimizations for large datasets
- **Analytics Charts**: 
  - Event type distribution
  - Work zone duration analysis
  - Update vs start date patterns

![Analytics Dashboard](images/analytics.png)
*Comprehensive analytics including event distributions, duration analysis, and temporal patterns*

## Project Structure

```
wzdx/
├── database/                     # Database files and data collection scripts
│   ├── az511.db                 # SQLite database for AZ511 work zones
│   ├── az511.py                 # AZ511 data collection script
│   ├── tomtom.db                # SQLite database for TomTom traffic flow
│   ├── tomtom.py                # TomTom traffic flow collection script
│   ├── workzones.db             # General work zone database
│   └── wzdx.py                  # Work zone data processing script
├── dashboard/                    # Streamlit dashboard applications
│   ├── az511app.py              # Main transportation dashboard
│   └── wzdxapp.py               # Work zone specific dashboard
├── inrix/                       # Legacy structure (may contain additional utilities)
├── .env                         # Environment variables (API keys)
├── .env.template                # Template for environment variables
├── .gitignore                   # Git ignore file (excludes .db files and .env)
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── setup.py                    # Package setup configuration
```

## Database Schemas

### AZ511 Database (`database/az511.db`)
```sql
TABLE events (
    ID, Organization, RoadwayName, DirectionOfTravel,
    Description, Reported, LastUpdated, StartDate,
    PlannedEndDate, LanesAffected, Latitude, Longitude,
    EventType, IsFullClosure, Severity
)
```

### TomTom Database (`database/tomtom.db`)
```sql
TABLE traffic_flow (
    id INTEGER PRIMARY KEY,
    segment_hash TEXT UNIQUE,        -- SHA256 hash for deduplication
    frc TEXT,                        -- Functional Road Class
    currentSpeed REAL,               -- Current speed (mph)
    freeFlowSpeed REAL,              -- Free flow speed (mph)
    currentTravelTime INTEGER,       -- Current travel time (seconds)
    freeFlowTravelTime INTEGER,      -- Free flow travel time (seconds)
    confidence REAL,                 -- Data confidence level
    roadClosure BOOLEAN,             -- Road closure status
    coordinates TEXT,                -- JSON array of lat/lon points
    version TEXT,                    -- API version
    coordinate_lat REAL,             -- Primary coordinate latitude
    coordinate_lon REAL,             -- Primary coordinate longitude
    timestamp INTEGER                -- Collection timestamp
)
```

## Data Flow

### TomTom Traffic Flow Data
1. **Collection**: `database/tomtom.py` queries TomTom API across Phoenix area grid
2. **Processing**: Creates unique hashes based on coordinates + FRC + timestamp
3. **Storage**: Batch inserts (every 100 API calls) into `database/tomtom.db`
4. **Visualization**: Dashboard displays traffic flow as colored lines on map with 5-tier speed system

### AZ511 Work Zone Data
1. **Collection**: `database/az511.py` fetches work zone events from AZ511 API
2. **Processing**: Handles datetime conversions and data validation
3. **Storage**: Stores events in `database/az511.db` with full temporal information
4. **Visualization**: Dashboard shows events as markers with analytics charts

## API Configuration

### TomTom Traffic API
- **Coverage**: Phoenix metropolitan area (33.2-33.7 lat, -112.4 to -111.8 lon)
- **Grid Resolution**: 0.01 degree increments for comprehensive coverage
- **Batch Processing**: Inserts every 100 API calls for efficiency
- **Rate Limiting**: 0.1 second delay between API calls
- **Deduplication**: SHA256 hash based on coordinates + FRC + rounded timestamp

### AZ511 API
- **Coverage**: Statewide Arizona work zones and incidents
- **Event Types**: Construction, accidents, closures, special events
- **Update Frequency**: Real-time as events are reported
- **Data Retention**: Historical events with start/end dates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License