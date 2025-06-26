"""
TomTom Traffic Flow API: https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data
output segment fields:dict_keys(['frc', 'currentSpeed', 'freeFlowSpeed', 'currentTravelTime', 'freeFlowTravelTime', 'confidence', 'roadClosure', 'coordinates', '@version'])
TODO:
1. Fetch only highway segments to limit API requests
    F unctional R oad C lass. This indicates the road type:

    FRC0 : Motorway, freeway or other major road
    FRC1 : Major road, less important than a motorway
    FRC2 : Other major road
    FRC3 : Secondary road
    FRC4 : Local connecting road
    FRC5 : Local road of high importance
    FRC6 : Local road
2. schedule this script to run periodically (e.g., hourly) to keep data fresh
"""
import os
import requests
import sqlite3
from dotenv import load_dotenv
import json
from datetime import datetime
import time
import numpy as np
import hashlib
from shapely.geometry import Polygon, Point

# Load environment variables
load_dotenv()

# Define polygon coordinates for in AZ area
# https://www.google.com/maps/d/edit?hl=en&mid=193Kg54jEnpC6mYp4FSXlNTpSC7Vzfhs&ll=33.60345294480069%2C-112.12479300115264&z=11
polygon_coords = [
    (-112.2796304, 33.6783582),
    (-112.2816903, 33.3887436),
    (-112.1244485, 33.2562061),
    (-111.5977914, 33.2642445),
    (-111.6108377, 33.4872985),
    (-111.8724496, 33.4872985),
    (-111.8786294, 33.6857861),
    (-112.2796304, 33.6783582)
]
polygon = Polygon(polygon_coords)
DEGREE_STEP = 0.01  # Step size for latitude/longitude grid in degrees, 0.01~= 1km at equator

def create_database():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Since tomtom.py is now in the database directory, the db file is in the same directory
    db_path = os.path.join(script_dir, 'tomtom.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create traffic_flow table matching TomTom API response schema exactly
    c.execute('''
        CREATE TABLE IF NOT EXISTS traffic_flow (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_hash TEXT UNIQUE,
            frc TEXT,
            currentSpeed REAL,
            freeFlowSpeed REAL,
            currentTravelTime INTEGER,
            freeFlowTravelTime INTEGER,
            confidence REAL,
            roadClosure BOOLEAN,
            coordinates TEXT,
            version TEXT,
            coordinate_lat REAL,
            coordinate_lon REAL,
            timestamp INTEGER
        )
    ''')
    conn.commit()
    return conn

def fetch_tomtom_traffic_flow(api_key, max_zoom=15, sample_only=True):
    """
    Fetch traffic flow data from TomTom API for Gilbert, AZ area
    URL format: https://{baseURL}/traffic/services/{versionNumber}/flowSegmentData/{style}/{zoom}/{format}
    """
    base_url = 'https://api.tomtom.com'
    version = '4'
    style = 'absolute'  # or 'relative'
    format_type = 'json'
    
    url = f'{base_url}/traffic/services/{version}/flowSegmentData/{style}/{max_zoom}/{format_type}'
    
    # Gilbert, AZ center point - API requires point parameter, not bbox
    gilbert_center = "33.367,-111.759"  # lat,lon format for Gilbert, AZ center
    
    params = {
        'key': api_key,
        'point': gilbert_center,  # API requires point parameter
        'unit': 'mph',
        'thickness': 1,  # Smaller thickness to get more segments
        'openLr': False,
        'jsonp': None
    }
    
    try:
        print(f"Making TomTom API request to: {url}")
        print(f"Point: {gilbert_center} (Gilbert, AZ center)")
        print(f"Zoom level: {max_zoom}")
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if sample_only:
            # Print sample of the response structure
            print(f"Response keys: {list(data.keys())}")
            if 'flowSegmentData' in data:
                flow_data = data['flowSegmentData']
                print(f"Flow data type: {type(flow_data)}")
                
                # Check if flow_data is a list or dict
                if isinstance(flow_data, list):
                    print(f"Number of flow segments: {len(flow_data)}")
                    # Print first few segments as sample
                    for i, segment in enumerate(flow_data[:3]):  # Only first 3 segments
                        print(f"\nSample segment {i+1}:")
                        print(f"  Current Speed: {segment.get('currentSpeed', 'N/A')} mph")
                        print(f"  Free Flow Speed: {segment.get('freeFlowSpeed', 'N/A')} mph")
                        print(f"  Current Travel Time: {segment.get('currentTravelTime', 'N/A')} sec")
                        print(f"  Confidence: {segment.get('confidence', 'N/A')}")
                        print(f"  Road Closure: {segment.get('roadClosure', 'N/A')}")
                        coords = segment.get('coordinates', {})
                        coord_list = coords.get('coordinate', []) if isinstance(coords, dict) else []
                        print(f"  Coordinates: {len(coord_list)} points")
                elif isinstance(flow_data, dict):
                    print(f"Flow data is a dict with keys: {list(flow_data.keys())}")
                    # If it's a dict, it might contain the actual segments in a sub-key
                    for key, value in flow_data.items():
                        print(f"  {key}: {type(value)} - {value if not isinstance(value, (list, dict)) else f'Length: {len(value)}'}")
                else:
                    print(f"Unexpected flow_data type: {type(flow_data)}")
                    print(f"Flow data content: {flow_data}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching TomTom traffic flow data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def fetch_tomtom_traffic_flow_multiple_points(api_key, max_zoom=10, conn=None, batch_size=100):
    """
    Fetch traffic flow data from multiple points across Phoenix comprehensive coverage
    Performs batch inserts every batch_size API calls for memory efficiency
    """
    # Get bounding box from polygon
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    # Step 3: Create grid at degree intervals
    lats = np.arange(min_lat, max_lat, DEGREE_STEP)
    lons = np.arange(min_lon, max_lon, DEGREE_STEP)

    total_pts = len(lats) * len(lons)
    cnt = 0
    batch_segments = []
    total_inserted = 0
    base_url = 'https://api.tomtom.com'
    version = '4'
    style = 'absolute'
    format_type = 'json'

    for lat in lats:
        for lon in lons:
            point = f"{lat},{lon}"
            pt = Point(lon, lat)
            if polygon.contains(pt):
                url = f'{base_url}/traffic/services/{version}/flowSegmentData/{style}/{max_zoom}/{format_type}'
                params = {
                    'key': api_key,
                    'point': point,
                    'unit': 'mph',
                    'thickness': 10,
                    'openLr': False,
                    'jsonp': None
                }
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    if 'flowSegmentData' in data:
                        segment_data = data['flowSegmentData']
                        batch_segments.append(segment_data)

                except requests.exceptions.RequestException as e:
                    pass  # Continue with next point
                    
            cnt += 1
            time.sleep(0.1)  # Respect API rate limits
            
            # Batch insert every batch_size API calls
            if cnt % batch_size == 0 and batch_segments and conn:
                batch_data = {'flowSegmentData': batch_segments}
                inserted = insert_traffic_flow(conn, batch_data)
                total_inserted += inserted
                print(f"\nBatch {cnt//batch_size}: Inserted {inserted} segments. Total inserted: {total_inserted}")
                batch_segments = []  # Clear batch
            
            print(f"{cnt}/{total_pts} - Point: {point}, Batch size: {len(batch_segments)}, Total inserted: {total_inserted}", end='\r')
    
    # Insert remaining segments in final batch
    if batch_segments and conn:
        batch_data = {'flowSegmentData': batch_segments}
        inserted = insert_traffic_flow(conn, batch_data)
        total_inserted += inserted
        print(f"\nFinal batch: Inserted {inserted} segments. Total inserted: {total_inserted}")
    
    print(f"\nTotal segments processed from all points: {cnt}")
    print(f"Total segments inserted to database: {total_inserted}")
    
    return total_inserted

def insert_traffic_flow(conn, flow_data):
    """Insert traffic flow data into database"""
    c = conn.cursor()
    inserted = 0
    current_timestamp = int(datetime.now().timestamp())
    
    if not flow_data or 'flowSegmentData' not in flow_data:
        return inserted
    
    # Handle the case where flowSegmentData is a single dict (one segment)
    segment_data = flow_data['flowSegmentData']
    
    if isinstance(segment_data, dict):
        # Single segment case
        segments = [segment_data]
    elif isinstance(segment_data, list):
        # Multiple segments case
        segments = segment_data
    else:
        print(f"Unexpected flowSegmentData type: {type(segment_data)}")
        return inserted
    
    for segment in segments:
        # Get coordinates
        coordinates = segment.get('coordinates', {})
        
        # Handle different coordinate formats
        coord_list = []
        if isinstance(coordinates, dict):
            coord_list = coordinates.get('coordinate', [])
        elif isinstance(coordinates, list):
            coord_list = coordinates
        
        if not coord_list:
            print(f"No coordinates found for segment: {segment}")
            continue
            
        first_coord = coord_list[0]
        
        # Create unique identifier for this segment using SHA256 hash
        # Round timestamp to nearest minute for consistent hashing
        rounded_timestamp = (current_timestamp // 60) * 60
        
        # Create hash string from coordinates + frc + rounded timestamp
        hash_string = f"{json.dumps(coord_list, sort_keys=True)}_{segment.get('frc', '')}_{rounded_timestamp}"
        segment_hash = hashlib.sha256(hash_string.encode()).hexdigest()
        
        # Check if this segment hash already exists
        existing = c.execute('''
            SELECT COUNT(*) FROM traffic_flow WHERE segment_hash = ?
        ''', (segment_hash,)).fetchone()[0]
        
        if existing > 0:
            continue  # Skip duplicate
        
        flow_record = (
            segment_hash,  # segment_hash for unique identification
            str(segment.get('frc', '')),  # frc as string (e.g., 'FRC4')
            float(segment.get('currentSpeed', 0)) if segment.get('currentSpeed') else None,
            float(segment.get('freeFlowSpeed', 0)) if segment.get('freeFlowSpeed') else None,
            int(segment.get('currentTravelTime', 0)) if segment.get('currentTravelTime') else None,
            int(segment.get('freeFlowTravelTime', 0)) if segment.get('freeFlowTravelTime') else None,
            float(segment.get('confidence', 0)) if segment.get('confidence') else None,
            1 if segment.get('roadClosure') else 0,
            json.dumps(coord_list),  # Store all coordinates as JSON
            str(segment.get('@version', '')),  # version field
            float(first_coord.get('latitude', 0)),  # coordinate_lat for convenience
            float(first_coord.get('longitude', 0)),  # coordinate_lon for convenience
            current_timestamp
        )
        
        c.execute('''
            INSERT INTO traffic_flow (
                segment_hash, frc, currentSpeed, freeFlowSpeed, currentTravelTime, 
                freeFlowTravelTime, confidence, roadClosure, coordinates,
                version, coordinate_lat, coordinate_lon, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', flow_record)
        inserted += 1
    
    conn.commit()
    return inserted

def main():
    api_key = os.getenv('TOMTOM_API_KEY')
    if not api_key:
        print("Error: TOMTOM_API_KEY not found in environment variables")
        return
    
    print("Starting TomTom traffic data collection for Gilbert, AZ")
    print("=" * 60)
    
    # Create or connect to database
    conn = create_database()
    
    # Clear existing data to avoid duplicates
    c = conn.cursor()
    c.execute("DELETE FROM traffic_flow")
    conn.commit()
    print("Cleared existing traffic flow data")
    
    print(f"Fetching traffic data for Phoenix area...")
    print(f"Using multiple points for comprehensive coverage with batch processing")
    print("Collecting traffic flow segments with batch inserts every 100 API calls.\n")
    
    # Fetch and insert data in batches
    total_inserted = fetch_tomtom_traffic_flow_multiple_points(api_key, max_zoom=10, conn=conn, batch_size=100)
        

    
    print(f"\nTotal records inserted: {total_inserted}")
    print("\n" + "-" * 40 + "\n")
    

    # Show database summary
    c = conn.cursor()
    flow_count = c.execute("SELECT COUNT(*) FROM traffic_flow").fetchone()[0]
    
    print(f"\n" + "=" * 60)
    print(f"Database Summary:")
    print(f"  Total traffic flow records: {flow_count}")
    print(f"  Database: tomtom.db")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    conn.close()

if __name__ == "__main__":
    main()
