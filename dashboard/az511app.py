"""
Streamlit dashboard showing AZ511 work zones on a map
Run: streamlit run dashboard/az511app.py
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

# Set page config to wide mode
st.set_page_config(layout="wide")

CITY_COORDS = {
    'Phoenix': {'lat': 33.4484, 'lon': -112.0740, 'zoom': 11},
    'Tucson': {'lat': 32.2226, 'lon': -110.9747, 'zoom': 11},
    'Flagstaff': {'lat': 35.1983, 'lon': -111.6513, 'zoom': 11},
    'Gilbert': {'lat': 33.3528, 'lon': -111.7890, 'zoom': 11},
    'Yuma': {'lat': 32.6927, 'lon': -114.6277, 'zoom': 11}
}

# Define color map for event types
color_map = {
    'roadwork': '#1f77b4',  # blue
    'specialEvents': '#9467bd',     # purple
    'accidentsAndIncidents': "#f70000",  # red
    'closures': "#ffe30e",       # yellow
    'traffic_flow': '#00ff00'    # green for traffic flow
}
event_type_size = {
    'roadwork': 0.2,
    'specialEvents': 2,
    'accidentsAndIncidents': 5,
    'closures': 2,
    'traffic_flow': 1
}

class AZ511DB:
    def __init__(self, db_path="az511.db"):
        self.db_path = Path(__file__).parent.parent / "database" / db_path
        
    def get_active_events(self, start_date, end_date, city=None):
        """Get events active during the selected date range and city area"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT ID, Organization, RoadwayName, DirectionOfTravel,
                    Description, Reported, LastUpdated, StartDate,
                    PlannedEndDate, LanesAffected, Latitude, Longitude,
                    EventType, IsFullClosure, Severity
                FROM events 
                WHERE (
                    (EventType = 'accidentsAndIncidents' AND StartDate BETWEEN ? AND ?) OR
                    (EventType != 'accidentsAndIncidents' AND StartDate <= ? AND PlannedEndDate >= ?)
                )
            """
            params = [int(start_date.timestamp()), int(end_date.timestamp()), int(end_date.timestamp()), int(start_date.timestamp())]
            
            if city and city in CITY_COORDS:
                # Add 0.1 degree radius around city center (roughly 11km)
                center = CITY_COORDS[city]
                query += """
                    AND Latitude BETWEEN ? AND ?
                    AND Longitude BETWEEN ? AND ?
                """
                params.extend([
                    center['lat'] - 0.1, center['lat'] + 0.1,
                    center['lon'] - 0.1, center['lon'] + 0.1
                ])
            
            query += " ORDER BY LastUpdated DESC"
            return conn.execute(query, params).fetchall()

    def get_daily_counts(self, start_date, end_date):
        """Get daily count of active events"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                WITH RECURSIVE dates(date) AS (
                    SELECT ?
                    UNION ALL
                    SELECT date + 86400
                    FROM dates
                    WHERE date < ?
                )
                SELECT 
                    dates.date,
                    COUNT(DISTINCT events.ID) as event_count
                FROM dates
                LEFT JOIN events ON 
                    dates.date >= events.StartDate AND
                    dates.date <= events.PlannedEndDate
                GROUP BY dates.date
                ORDER BY dates.date
            """
            return conn.execute(query, (
                int(start_date.timestamp()),
                int(end_date.timestamp())
            )).fetchall()

class TomTomDB:
    def __init__(self, db_path="tomtom.db"):
        self.db_path = Path(__file__).parent.parent / "database" / db_path
        
    def get_traffic_flow_segments(self, city=None, frc_filter=None, limit=None):
        """Get traffic flow segments from TomTom data, filtered by FRC (road class)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT 
                    td.id, 
                    rs.coordinate_lat, 
                    rs.coordinate_lon, 
                    td.currentSpeed, 
                    td.freeFlowSpeed, 
                    td.currentTravelTime, 
                    td.freeFlowTravelTime,
                    td.confidence, 
                    td.roadClosure, 
                    rs.coordinates, 
                    td.timestamp, 
                    rs.frc, 
                    td.version,
                    rs.segment_id,
                    rs.openlr
                FROM traffic_data td
                INNER JOIN road_segments rs ON td.segment_id = rs.segment_id
            """
            params = []
            conditions = []
            
            # Default to FRC0 and FRC1 (major highways) if no filter specified
            if frc_filter is None:
                frc_filter = ['FRC0', 'FRC1']
            
            if frc_filter:
                placeholders = ','.join('?' * len(frc_filter))
                conditions.append(f"rs.frc IN ({placeholders})")
                params.extend(frc_filter)
            
            if city and city in CITY_COORDS:
                # Add 0.1 degree radius around city center (roughly 11km)
                center = CITY_COORDS[city]
                conditions.extend([
                    "rs.coordinate_lat BETWEEN ? AND ?",
                    "rs.coordinate_lon BETWEEN ? AND ?"
                ])
                params.extend([
                    center['lat'] - 0.1, center['lat'] + 0.1,
                    center['lon'] - 0.1, center['lon'] + 0.1
                ])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Prioritize higher importance roads (FRC0, FRC1) and limit results for performance
            query += """ ORDER BY 
                CASE rs.frc 
                    WHEN 'FRC0' THEN 0 
                    WHEN 'FRC1' THEN 1 
                    WHEN 'FRC2' THEN 2 
                    WHEN 'FRC3' THEN 3 
                    WHEN 'FRC4' THEN 4 
                    WHEN 'FRC5' THEN 5 
                    WHEN 'FRC6' THEN 6 
                    ELSE 7 
                END, td.timestamp DESC, td.id
            """
            
            # Apply smart limiting based on FRC types to improve performance
            if limit:
                query += f" LIMIT {limit}"
            elif frc_filter and any(frc in ['FRC4', 'FRC5', 'FRC6'] for frc in frc_filter):
                # Limit local roads to prevent performance issues
                query += " LIMIT 1000"
            
            return conn.execute(query, params).fetchall()
    
    def get_database_summary(self):
        """Get summary statistics for TomTom database"""
        with sqlite3.connect(self.db_path) as conn:
            # Get total number of traffic data records and road segments
            traffic_count = pd.read_sql_query("SELECT COUNT(*) as count FROM traffic_data", conn).iloc[0]['count']
            segments_count = pd.read_sql_query("SELECT COUNT(*) as count FROM road_segments", conn).iloc[0]['count']
            
            # Get FRC distribution from road_segments
            frc_distribution = pd.read_sql_query("SELECT frc, COUNT(*) as count FROM road_segments GROUP BY frc ORDER BY frc", conn)
            
            # Get time range from traffic_data
            if traffic_count > 0:
                time_range = pd.read_sql_query("""
                    SELECT 
                        datetime(MIN(timestamp), 'unixepoch') as earliest,
                        datetime(MAX(timestamp), 'unixepoch') as latest 
                    FROM traffic_data
                """, conn)
                earliest = time_range['earliest'].iloc[0]
                latest = time_range['latest'].iloc[0]
            else:
                earliest = None
                latest = None
            
            return {
                'total_flow': traffic_count,
                'total_segments': segments_count,
                'frc_distribution': frc_distribution,
                'earliest': earliest,
                'latest': latest
            }

def add_optimized_traffic_flow(fig, tomtom_data):
    """Optimized function to add TomTom traffic flow data to map"""
    if tomtom_data.empty:
        return
    
    # Pre-calculate all coordinates and colors to minimize processing in loop
    segments_by_frc = {}
    
    # Group segments by FRC for batch processing
    for frc in tomtom_data['frc'].unique():
        frc_data = tomtom_data[tomtom_data['frc'] == frc]
        segments_by_frc[frc] = []
        
        for idx, segment in frc_data.iterrows():
            try:
                # Parse coordinates from JSON string
                coordinates = json.loads(segment['coordinates'])
                
                if len(coordinates) >= 2:
                    # Simplify coordinates for performance (take every nth point for long segments)
                    if len(coordinates) > 10:
                        # For long segments, sample every 3rd point to reduce complexity
                        coordinates = coordinates[::3]
                    
                    # Extract lat/lon for the line
                    lats = [coord['latitude'] for coord in coordinates]
                    lons = [coord['longitude'] for coord in coordinates]
                    
                    # Calculate speed ratio and color
                    current_speed = segment.get('currentSpeed', 0)
                    free_flow_speed = segment.get('freeFlowSpeed', 1)
                    
                    if free_flow_speed > 0:
                        speed_ratio = max(0.0, min(1.0, current_speed / free_flow_speed))
                    else:
                        speed_ratio = 0.5
                    
                    # Use 5-tier color mapping for more granular traffic flow visualization
                    if speed_ratio > 0.9:
                        color = "#027a02"  # Green - excellent flow (90%+)
                    elif speed_ratio > 0.7:
                        color = '#7fff00'  # Light green - good flow (70-90%)
                    elif speed_ratio > 0.5:
                        color = '#ffff00'  # Yellow - moderate flow (50-70%)
                    elif speed_ratio > 0.3:
                        color = '#ff8c00'  # Orange - slow flow (30-50%)
                    else:
                        color = '#ff0000'  # Red - very slow/stopped flow (<30%)
                    
                    speed_text = f"{frc}: {current_speed:.1f}/{free_flow_speed:.1f} mph"
                    
                    segments_by_frc[frc].append({
                        'lats': lats,
                        'lons': lons,
                        'color': color,
                        'speed_text': speed_text,
                        'speed_ratio': speed_ratio
                    })
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    
    # Add traces by FRC group for better performance and legend organization
    for frc, segments in segments_by_frc.items():
        if not segments:
            continue
            
        # Determine line width based on FRC importance
        line_width = {
            'FRC0': 6, 'FRC1': 5, 'FRC2': 4, 
            'FRC3': 3, 'FRC4': 2, 'FRC5': 2, 'FRC6': 1
        }.get(frc, 2)
        
        # Batch similar colored segments together for performance
        color_groups = {}
        for segment in segments:
            color = segment['color']
            if color not in color_groups:
                color_groups[color] = {'lats': [], 'lons': [], 'texts': []}
            
            # Add None values to separate line segments
            color_groups[color]['lats'].extend(segment['lats'] + [None])
            color_groups[color]['lons'].extend(segment['lons'] + [None])
            color_groups[color]['texts'].extend([segment['speed_text']] * len(segment['lats']) + [None])
        
        # Create one trace per color per FRC for optimal performance
        for i, (color, data) in enumerate(color_groups.items()):
            show_legend = i == 0  # Only show legend for first trace of each FRC
            legend_name = f'Traffic {frc}' if show_legend else None
            
            fig.add_trace(go.Scattermap(
                lat=data['lats'],
                lon=data['lons'],
                mode='lines',
                line=dict(width=line_width, color=color),
                name=legend_name,
                text=data['texts'],
                hovertemplate="<b>Traffic Flow</b><br>%{text}<br><extra></extra>",
                showlegend=show_legend,
                legendgroup=frc  # Group all traces of same FRC together
            ))

def add_geojson_roads(fig):
    """Add Arizona road networks from GeoJSON files as separate layers"""
    database_dir = Path(__file__).parent.parent / "database"
    
    # Define GeoJSON files and their display properties
    geojson_files = {
        'az_interstates.geojson': {
            'name': 'Arizona Interstates',
            'color': '#2E86AB',  # Blue
            'width': 4,
            'opacity': 0.8
        },
        'az_sr.geojson': {
            'name': 'Arizona State Routes', 
            'color': '#A23B72',  # Purple
            'width': 2,
            'opacity': 0.6
        }
    }
    
    for filename, properties in geojson_files.items():
        geojson_path = database_dir / filename
        
        try:
            if geojson_path.exists():
                with open(geojson_path, 'r') as f:
                    geojson_data = json.load(f)
                
                # Extract coordinates from all LineString features
                all_lats = []
                all_lons = []
                
                for feature in geojson_data['features']:
                    if feature['geometry']['type'] == 'LineString':
                        coords = feature['geometry']['coordinates']
                        # GeoJSON coordinates are [longitude, latitude]
                        lons = [coord[0] for coord in coords]
                        lats = [coord[1] for coord in coords]
                        
                        # Add coordinates to lists with None separator for multiple lines
                        all_lons.extend(lons + [None])
                        all_lats.extend(lats + [None])
                
                # Add the road network as a single trace
                if all_lats and all_lons:
                    fig.add_trace(go.Scattermap(
                        lat=all_lats,
                        lon=all_lons,
                        mode='lines',
                        line=dict(
                            width=properties['width'],
                            color=properties['color']
                        ),
                        opacity=properties['opacity'],
                        name=properties['name'],
                        hovertemplate=f"<b>{properties['name']}</b><br>" +
                                    "Location: (%{lat:.4f}, %{lon:.4f})<extra></extra>",
                        showlegend=True,
                        legendgroup='roads'  # Group road layers together
                    ))
                    
        except Exception as e:
            st.warning(f"Could not load {filename}: {str(e)}")
            continue

def main():
    st.title("Arizona Transportation Dashboard")
    
    # Initialize database connections
    az511_db = AZ511DB()
    tomtom_db = TomTomDB()
    
    # Add data source and filter options in sidebar
    st.sidebar.subheader("Data Sources")
    
    # Data source selection with checkboxes
    show_az511 = st.sidebar.checkbox("Show AZ511 Work Zones", value=True, key="show_az511")
    show_tomtom = st.sidebar.checkbox("Show TomTom Traffic Flow", value=True, key="show_tomtom")
    show_roads = st.sidebar.checkbox("Show Arizona Road Networks", value=True, key="show_roads")
    
    # TomTom FRC (road class) filter - only show if TomTom is enabled
    frc_filter = None
    if show_tomtom:
        st.sidebar.write("**TomTom Road Types:**")
        with st.sidebar.expander("ℹ️ About Road Classifications"):
            st.write("""
            **FRC (Functional Road Class)** indicates road importance:
            - **FRC0**: Motorways/Freeways (highest priority)
            - **FRC1**: Major arterial roads  
            - **FRC2**: Other major roads
            - **FRC3**: Secondary roads
            - **FRC4**: Local connecting roads
            - **FRC5**: Local roads (high importance)
            - **FRC6**: Local roads (lowest priority)
            
            *Default: FRC0 & FRC1 (major highways only)*
            """)
        
        frc_options = {
            'FRC0': 'Motorways/Freeways (FRC0)',
            'FRC1': 'Major Roads (FRC1)', 
            'FRC2': 'Other Major Roads (FRC2)',
            'FRC3': 'Secondary Roads (FRC3)',
            'FRC4': 'Local Connecting Roads (FRC4)',
            'FRC5': 'Local Roads - High Importance (FRC5)',
            'FRC6': 'Local Roads (FRC6)'
        }
        
        selected_frcs = []
        # Default to FRC0 and FRC1 selected
        for frc_code, frc_desc in frc_options.items():
            default_selected = frc_code in ['FRC0', 'FRC1']
            if st.sidebar.checkbox(frc_desc, value=default_selected, key=f"frc_{frc_code}"):
                selected_frcs.append(frc_code)
        
        # Add performance warning for local roads
        if any(frc in selected_frcs for frc in ['FRC4', 'FRC5', 'FRC6']):
            st.sidebar.warning("⚠️ Local roads (FRC4+) may slow down map rendering. Limited to 1000 segments for performance.")
        
        frc_filter = selected_frcs if selected_frcs else None
    
    # Road networks info - only show if roads are enabled  
    if show_roads:
        st.sidebar.write("**Road Networks:**")
        with st.sidebar.expander("ℹ️ About Road Networks"):
            st.write("""
            **Arizona Road Networks** from GeoJSON data:
            - **Interstate Highways** (blue) - Major interstate routes
            - **State Routes** (purple) - Arizona state highways
            
            *Road data sourced from Arizona Department of Transportation*
            """)
    
    st.sidebar.subheader("Filter Options")
    
    selected_city = st.sidebar.selectbox(
        "Select City",
        ["All Cities"] + list(CITY_COORDS.keys()),
        key="city_selector"
    )
    # Get default date range (today only)
    today = datetime.now().date()

    # Add a slider for selecting the date
    selected_date = st.slider(
        "Select Date",
        min_value=today - timedelta(days=15),
        max_value=today,
        value=today,
        format="MM/DD/YY",
        key="selected_date"
    )

    # Convert selected date to datetime
    start_str = datetime.combine(selected_date, datetime.min.time())
    end_str = datetime.combine(selected_date, datetime.max.time())
    
    # Get active events with city filter
    city_filter = None if selected_city == "All Cities" else selected_city
    
    # Prepare data for visualization
    all_data = []
    
    # Get AZ511 data if selected
    if show_az511:
        events = az511_db.get_active_events(
            start_str,
            end_str,
            city_filter
        )
        
        if events:
            # Convert to DataFrame for plotting
            df_az511 = pd.DataFrame([dict(row) for row in events])
            df_az511['data_source'] = 'AZ511'
            all_data.append(df_az511)
    
    # Get TomTom traffic flow data if selected
    if show_tomtom:
        traffic_segments = tomtom_db.get_traffic_flow_segments(city_filter, frc_filter)
        
        if traffic_segments:
            # Convert to DataFrame for plotting
            df_tomtom = pd.DataFrame([dict(row) for row in traffic_segments])
            
            # Create description for hover text based on traffic data
            df_tomtom['Description'] = df_tomtom.apply(lambda x: 
                f"Traffic Flow ({x['frc']}) - Speed: {x['currentSpeed']:.1f}/{x['freeFlowSpeed']:.1f} mph", axis=1)
            df_tomtom['EventType'] = 'traffic_flow'
            df_tomtom['Organization'] = 'TomTom'
            df_tomtom['RoadwayName'] = df_tomtom['frc'].apply(lambda x: f"{x} Road" if pd.notna(x) else 'Unknown Road')
            df_tomtom['LastUpdated'] = pd.to_datetime(df_tomtom['timestamp'], unit='s')
            df_tomtom['StartDate'] = df_tomtom['LastUpdated']  # Use timestamp as start date
            df_tomtom['PlannedEndDate'] = df_tomtom['LastUpdated']  # Same as start for traffic flow
            df_tomtom['LanesAffected'] = 'N/A'
            df_tomtom['Severity'] = df_tomtom['confidence'].apply(lambda x: f"Confidence: {x:.2f}" if pd.notna(x) else "N/A")
            df_tomtom['Latitude'] = df_tomtom['coordinate_lat']
            df_tomtom['Longitude'] = df_tomtom['coordinate_lon']
            df_tomtom['data_source'] = 'TomTom'
            all_data.append(df_tomtom)
    
    # Combine all data sources
    if all_data:
        df = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Handle datetime conversions with error checking (only for AZ511 data)
        if show_az511:
            az511_mask = df['data_source'] == 'AZ511'
            
            max_timestamp = pd.Timestamp.max.timestamp()
            min_timestamp = pd.Timestamp.min.timestamp()
            
            def safe_to_datetime(timestamp):
                try:
                    if timestamp is None:
                        return None
                    # Check if timestamp is within valid range
                    if timestamp > max_timestamp or timestamp < min_timestamp:
                        # Return a timezone-aware max timestamp in MST
                        mst_offset = timezone(timedelta(hours=-7))
                        return pd.Timestamp.max.tz_localize('UTC').tz_convert(mst_offset)
                    dt = pd.to_datetime(timestamp, unit='s')
                    # Calculate the MST offset (UTC-7)
                    mst_offset = timezone(timedelta(hours=-7))
                    # Localize the datetime object to MST
                    mst_time = dt.replace(tzinfo=timezone.utc).astimezone(mst_offset)
                    return mst_time
                except Exception:
                    # Return a timezone-aware max timestamp in MST
                    mst_offset = timezone(timedelta(hours=-7))
                    return pd.Timestamp.max.tz_localize('UTC').tz_convert(mst_offset)
            
            # Safely convert timestamps to datetime for AZ511 data only
            if az511_mask.any():
                df.loc[az511_mask, 'StartDate'] = df.loc[az511_mask, 'StartDate'].apply(safe_to_datetime)
                df.loc[az511_mask, 'PlannedEndDate'] = df.loc[az511_mask, 'PlannedEndDate'].apply(safe_to_datetime)
                df.loc[az511_mask, 'LastUpdated'] = df.loc[az511_mask, 'LastUpdated'].apply(safe_to_datetime)
                # if 'Reported' in df.columns:
                #     df.loc[az511_mask, 'Reported'] = df.loc[az511_mask, 'Reported'].apply(safe_to_datetime)

            # Filter out events with invalid dates (None values or max timestamps) - only for AZ511
            mst_offset = timezone(timedelta(hours=-7))
            max_timestamp_tz = pd.Timestamp.max.tz_localize('UTC').tz_convert(mst_offset)
            
            # For incidents: only require valid StartDate
            # For other events: require both valid StartDate and PlannedEndDate
            incidents_mask = (df['EventType'] == 'accidentsAndIncidents') & az511_mask
            other_events_mask = (df['EventType'] != 'accidentsAndIncidents') & (df['EventType'] != 'traffic_flow') & az511_mask
            traffic_flow_mask = df['EventType'] == 'traffic_flow'
            
            valid_incidents = (
                incidents_mask &
                df['StartDate'].notna() &
                (df['StartDate'] != max_timestamp_tz)
            )
            
            valid_other_events = (
                other_events_mask &
                df['StartDate'].notna() & 
                df['PlannedEndDate'].notna() &
                (df['StartDate'] != max_timestamp_tz) &
                (df['PlannedEndDate'] != max_timestamp_tz)
            )
            
            # Traffic flow data is always valid (TomTom data is already processed)
            valid_traffic_flow = traffic_flow_mask
            
            valid_dates = valid_incidents | valid_other_events | valid_traffic_flow

            # Capture filtered events for display
            filtered_events = df[~valid_dates].copy() if not valid_dates.all() else pd.DataFrame()
            
            if not valid_dates.all():
                st.warning(f"Filtered out {(~valid_dates).sum()} events with invalid dates")
                df = df[valid_dates].copy()
        
        # Get map center and zoom based on selected city
        if city_filter:
            center = CITY_COORDS[city_filter]
            map_center = {"lat": center['lat'], "lon": center['lon']}
            map_zoom = center['zoom']
        else:
            map_center = {"lat": 33.4484, "lon": -112.0740}  # Phoenix as default
            map_zoom = 10  # Increased default zoom for better view
        
        # Create the map with both data sources
        fig = go.Figure()
        
        # Add an invisible marker at map center to ensure proper zoom/center even with no data
        # fig.add_trace(go.Scattermap(
        #     lat=[map_center["lat"]],
        #     lon=[map_center["lon"]],
        #     mode='markers',
        #     marker=dict(size=1, opacity=0),  # Invisible marker
        #     showlegend=False,
        #     hoverinfo='skip'
        # ))
        
        # Add AZ511 data as scatter points
        if show_az511 and 'data_source' in df.columns:
            az511_data = df[df['data_source'] == 'AZ511']
            if not az511_data.empty:
                fig = px.scatter_map(
                    az511_data,
                    lat="Latitude",
                    lon="Longitude",
                    color="EventType",
                    size=az511_data['EventType'].map(event_type_size),
                    hover_name="Description",
                    hover_data=["RoadwayName", "Severity"],
                    zoom=map_zoom,
                    center=map_center,
                    height=600,
                    color_discrete_map=color_map
                )

                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":30,"t":30,"l":30,"b":50},
                    title=f"AZ511 Events - {selected_city} ({selected_date})"
                )

        # Add TomTom traffic flow data as lines (optimized)
        if show_tomtom and 'data_source' in df.columns:
            tomtom_data = df[df['data_source'] == 'TomTom']
            if not tomtom_data.empty:
                # Add progress indicator
                with st.spinner(f'Rendering {len(tomtom_data)} traffic segments...'):
                    add_optimized_traffic_flow(fig, tomtom_data)
        
        # Add Arizona road networks from GeoJSON files
        if show_roads:
            add_geojson_roads(fig)
        
        # Update layout for mapbox - ensure it always centers on the specified location
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                # center=dict(lat=map_center["lat"], lon=map_center["lon"]),
                center=map_center,
                zoom=map_zoom
            ),
            margin={"r":180,"t":30,"l":30,"b":50},  # Increased right margin for legend
            title=f"Transportation Data - {selected_city} ({selected_date})",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,  # Position legend to the right of the map
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            height=600  # Set a fixed height to ensure proper rendering
        )
        
        # Display the map
        st.plotly_chart(fig, use_container_width=True)
        
        # Display work zone count and details
        az511_count = len(df[df['EventType'] != 'traffic_flow']) if not df.empty else 0
        tomtom_count = len(df[df['EventType'] == 'traffic_flow']) if not df.empty else 0
        
        # Add AZ511 analytics charts (only for AZ511 data)
        if show_az511:
            az511_data = df[df['EventType'] != 'traffic_flow'] if not df.empty else pd.DataFrame()
            if not az511_data.empty:
                st.subheader("AZ511 Analytics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Event Type Distribution
                    st.write("**Event Type Distribution**")
                    event_counts = az511_data['EventType'].value_counts()
                    if len(event_counts) > 0:
                        fig_pie = px.pie(
                            values=event_counts.values,
                            names=event_counts.index,
                            title="Work Zone Events by Type"
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("No event type data available")
                
                with col2:
                    # Duration distribution - only for AZ511 data with valid dates
                    az511_data_valid = az511_data[az511_data['EventType'] != 'accidentsAndIncidents'].copy()
                    
                    # Handle datetime conversion safely for duration calculation
                    if not az511_data_valid.empty:
                        try:
                            # Convert StartDate and PlannedEndDate to naive datetime for calculation
                            def convert_to_naive(dt_series):
                                """Convert datetime series to naive datetime, handling both aware and naive inputs"""
                                converted = pd.to_datetime(dt_series)
                                if converted.dt.tz is not None:
                                    return converted.dt.tz_localize(None)
                                return converted
                            
                            az511_data_valid['start_naive'] = convert_to_naive(az511_data_valid['StartDate'])
                            az511_data_valid['end_naive'] = convert_to_naive(az511_data_valid['PlannedEndDate'])
                            
                            # Filter out invalid end dates
                            valid_duration_mask = az511_data_valid['end_naive'].dt.year < 2263
                            az511_data_valid = az511_data_valid[valid_duration_mask].copy()
                            
                            if not az511_data_valid.empty:
                                # Calculate duration in days
                                az511_data_valid['duration_days'] = (az511_data_valid['end_naive'] - az511_data_valid['start_naive']).dt.total_seconds() / 3600 / 24
                                
                                # Cap duration at 90 days
                                az511_data_valid.loc[az511_data_valid['duration_days'] > 90, 'duration_days'] = 90
                                
                                fig_duration = px.histogram(
                                    az511_data_valid,
                                    x='duration_days',
                                    color='EventType',
                                    color_discrete_map=color_map,
                                    barmode='overlay',
                                    opacity=0.7,
                                    title="Work Zone Duration Distribution",
                                    labels={'duration_days': 'Duration (days)', 'EventType': 'Event Type'},
                                    nbins=30,
                                    range_x=[0, 90]
                                )
                                
                                # Add a vertical line at 90 days
                                fig_duration.add_vline(
                                    x=90, 
                                    line_dash="dash", 
                                    line_color="gray",
                                    annotation_text="events >90 days"
                                )
                                
                                fig_duration.update_layout(
                                    xaxis_title="Duration (days, capped at 90)",
                                    yaxis_title="Number of Events",
                                    legend_title_text="Event Type",
                                    bargap=0.1
                                )
                                st.plotly_chart(fig_duration, use_container_width=True)
                            else:
                                st.info("No events with valid duration found")
                        except Exception as e:
                            st.warning(f"Could not calculate duration: {str(e)}")
                    else:
                        st.info("No work zone events for duration analysis")
                

                col1, col2 = st.columns(2)
                with col1:
                    # Update date vs Start date scatter - only for AZ511 data
                    try:
                        az511_scatter_data = az511_data.copy()
                        
                        def convert_to_naive(dt_series):
                            """Convert datetime series to naive datetime, handling both aware and naive inputs"""
                            converted = pd.to_datetime(dt_series)
                            if converted.dt.tz is not None:
                                return converted.dt.tz_localize(None)
                            return converted
                        
                        az511_scatter_data['update_datetime'] = convert_to_naive(az511_scatter_data['LastUpdated'])
                        az511_scatter_data['start_datetime'] = convert_to_naive(az511_scatter_data['StartDate'])
                        
                        fig_dates = px.scatter(
                            az511_scatter_data,
                            x='update_datetime',
                            y='start_datetime',
                            color='EventType',
                            title="Update Date vs Start Date",
                            labels={
                                'start_datetime': 'Start Date',
                                'update_datetime': 'Update Date',
                                'EventType': 'Event Type'
                            },
                            color_discrete_map=color_map
                        )
                        
                        # Add diagonal line where start_date = update_date
                        fig_dates.add_scatter(
                            x=[az511_scatter_data['start_datetime'].min(), az511_scatter_data['start_datetime'].max()],
                            y=[az511_scatter_data['start_datetime'].min(), az511_scatter_data['start_datetime'].max()],
                            mode='lines',
                            line=dict(dash='dash', color='gray'),
                            name='Start = Update',
                            showlegend=True
                        )
                        
                        fig_dates.update_layout(
                            xaxis_title="Update Date",
                            yaxis_title="Start Date",
                            legend_title_text="Event Type"
                        )
                        st.plotly_chart(fig_dates, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create scatter plot: {str(e)}")

                with col2:
                    # Event Type Distribution pie chart - AZ511 only
                    az511_event_counts = az511_data['EventType'].value_counts()
                    if not az511_event_counts.empty:
                        fig_event_types = px.pie(
                            values=az511_event_counts.values,
                            names=az511_event_counts.index,
                            title="AZ511 Event Type Distribution",
                            color=az511_event_counts.index,
                            color_discrete_map=color_map
                        )
                        fig_event_types.update_traces(
                            textposition='inside', 
                            textinfo='percent+label'
                        )
                        st.plotly_chart(fig_event_types, use_container_width=True)
                    else:
                        st.info("No AZ511 events to display")




        # Add database summary section
        st.subheader("Database Summary")
        
        # Create summary statistics for both databases
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**AZ511 Database:**")
            if show_az511:
                with sqlite3.connect(az511_db.db_path) as conn:
                    # Get total number of events
                    total_events = pd.read_sql_query("SELECT COUNT(*) as count FROM events", conn).iloc[0]['count']
                    
                    # Get time range
                    time_range = pd.read_sql_query("""
                        SELECT 
                            date(MIN(CASE 
                                WHEN datetime(StartDate, 'unixepoch') > datetime('2000-01-01') 
                                THEN StartDate 
                            END), 'unixepoch') as earliest,
                            date(MAX(CASE 
                                WHEN datetime(PlannedEndDate, 'unixepoch') < datetime('2200-01-01') 
                                THEN PlannedEndDate 
                            END), 'unixepoch') as latest 
                        FROM events
                    """, conn)
                    
                    st.metric("Total Events", total_events)
                    st.metric("Date Range", 
                             f"{time_range['earliest'].iloc[0]} to {time_range['latest'].iloc[0]}")
            else:
                st.info("AZ511 data not selected")
        
        with col2:
            st.write("**TomTom Database:**")
            if show_tomtom:
                tomtom_summary = tomtom_db.get_database_summary()
                displayed_segments = len([row for row in traffic_segments]) if 'traffic_segments' in locals() and traffic_segments else 0
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Traffic Records", f"{tomtom_summary['total_flow']}")
                    st.metric("Road Segments", f"{tomtom_summary['total_segments']}")
                with col2b:
                    st.metric("Displayed", f"{displayed_segments}")
                    if frc_filter:
                        st.write(f"**Filtered:** {', '.join(frc_filter)}")
                    else:
                        st.write("**Default:** FRC0, FRC1")
                
                # Show FRC distribution
                if 'frc_distribution' in tomtom_summary and not tomtom_summary['frc_distribution'].empty:
                    st.write("**Road Class Distribution:**")
                    for _, row in tomtom_summary['frc_distribution'].iterrows():
                        st.write(f"  • {row['frc']}: {row['count']} segments")
                
                if tomtom_summary['earliest']:
                    st.write(f"**Data Range:** {tomtom_summary['earliest']} to {tomtom_summary['latest']}")
            else:
                st.info("TomTom data not selected")

        # Show data source breakdown
        total_records = len(df)
        st.subheader(f"Active Data ({total_records} total records: {az511_count} AZ511 events, {tomtom_count} traffic segments)")
        

        # Show detailed information if requested
        if show_az511:
            az511_data = df[df['data_source'] == 'AZ511'] if 'data_source' in df.columns else df
            if not az511_data.empty:
                with st.expander("AZ511 Data Details"):
                    st.dataframe(az511_data[['EventType', 'Description', 'RoadwayName', 'LastUpdated']].head(10))
        
        if show_tomtom:
            tomtom_data = df[df['data_source'] == 'TomTom'] if 'data_source' in df.columns else pd.DataFrame()
            if not tomtom_data.empty:
                with st.expander("TomTom Traffic Flow Details"):
                    st.dataframe(tomtom_data.head(10))

    else:
        # No AZ511 or TomTom data, but still show road networks if requested
        if show_roads:
            # Create a map with just the road networks
            if city_filter:
                center = CITY_COORDS[city_filter]
                map_center = {"lat": center['lat'], "lon": center['lon']}
                map_zoom = center['zoom']
            else:
                map_center = {"lat": 33.4484, "lon": -112.0740}  # Phoenix as default
                map_zoom = 10
            
            fig = go.Figure()
            add_geojson_roads(fig)
            
            # Update layout for mapbox
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=map_center,
                    zoom=map_zoom
                ),
                margin={"r":180,"t":30,"l":30,"b":50},  # Increased right margin for legend
                title=f"Arizona Road Networks - {selected_city} ({selected_date})",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,  # Position legend to the right of the map
                    bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                height=600
            )
            
            # Display the map
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for the selected sources and filters.")
            if not show_az511 and not show_tomtom:
                st.info("Please select at least one data source from the sidebar.")
            else:
                st.code("""
                # To populate databases, run:
                # AZ511: python inrix/dashboard/az511.py
                # TomTom: python inrix/dashboard/tomtom.py
                """)

if __name__ == "__main__":
    main()