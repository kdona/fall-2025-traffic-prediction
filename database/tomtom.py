"""
TomTom Traffic Flow API integration for Arizona road networks.
Samples points along polylines from az_interstates.geojson and az_sr.geojson
and collects traffic flow data using TomTom's flowSegmentData API.

Database Schema:
- road_segments: stable segment information (segment_id, frc, coordinates)
- traffic_data: time-varying traffic data (foreign key to road_segments)

Functional Road Classes (FRC):
FRC0: Motorway/freeway, FRC1: Major road, FRC2: Other major road
FRC3: Secondary road, FRC4: Local connecting road, FRC5-6: Local roads
"""
import os
import requests
import sqlite3
from dotenv import load_dotenv
import json
from datetime import datetime
import time
import hashlib
from shapely.geometry import LineString, Point

# Load environment variables
load_dotenv()

# Global API parameters - modify these to optimize API calls and coverage
DEGREE_STEP = 0.025  # Step size for sampling in degrees (~2.75km at Phoenix latitude)
API_ZOOM = 10       # TomTom API zoom level (8=broader, 12=detailed)
API_THICKNESS = 10   # API thickness parameter (higher=more parallel roads)
BATCH_SIZE = 50     # Number of API calls to batch before database insert

def load_geojson_polylines():
    """Load polylines from GeoJSON files and return sampled points for API calls"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    geojson_files = ['az_interstates.geojson', 'az_sr.geojson']
    all_sample_points = []
    
    for filename in geojson_files:
        geojson_path = os.path.join(script_dir, filename)
        
        try:
            if os.path.exists(geojson_path):
                with open(geojson_path, 'r') as f:
                    geojson_data = json.load(f)
                
                for feature_idx, feature in enumerate(geojson_data['features']):
                    if feature['geometry']['type'] == 'LineString':
                        coords = feature['geometry']['coordinates']
                        line_coords = [(coord[0], coord[1]) for coord in coords]
                        line = LineString(line_coords)
                        
                        # Sample points along lines longer than 100m
                        line_length = line.length
                        length_meters = line_length * 111000
                        
                        if length_meters > 100:
                            sample_distance_degrees = DEGREE_STEP
                            num_samples = max(2, int(line_length / sample_distance_degrees))
                            
                            for i in range(num_samples + 1):
                                fraction = i / num_samples if num_samples > 0 else 0
                                point = line.interpolate(fraction, normalized=True)
                                
                                all_sample_points.append({
                                    'lat': point.y,
                                    'lon': point.x,
                                    'source_file': filename,
                                    'feature_idx': feature_idx,
                                    'point_idx': i,
                                    'distance_along_line_km': (fraction * length_meters) / 1000
                                })
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return all_sample_points

def create_database():
    """Create SQLite database with normalized schema for road segments and traffic data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'tomtom.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create road_segments table for stable segment information
    c.execute('''
        CREATE TABLE IF NOT EXISTS road_segments (
            segment_id TEXT PRIMARY KEY,
            openlr TEXT,
            frc TEXT,
            coordinates TEXT,
            coordinate_lat REAL,
            coordinate_lon REAL,
            created_timestamp INTEGER
        )
    ''')
    
    # Create traffic_data table for time-varying traffic information
    c.execute('''
        CREATE TABLE IF NOT EXISTS traffic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id TEXT,
            hash_id TEXT UNIQUE,
            timestamp INTEGER,
            currentSpeed REAL,
            freeFlowSpeed REAL,
            currentTravelTime INTEGER,
            freeFlowTravelTime INTEGER,
            confidence REAL,
            roadClosure BOOLEAN,
            version TEXT,
            FOREIGN KEY (segment_id) REFERENCES road_segments(segment_id)
        )
    ''')
    
    # Create indexes for better query performance
    c.execute('''CREATE INDEX IF NOT EXISTS idx_traffic_data_segment_id ON traffic_data(segment_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_traffic_data_timestamp ON traffic_data(timestamp)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_traffic_data_hash_id ON traffic_data(hash_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_road_segments_frc ON road_segments(frc)''')
    
    conn.commit()
    return conn

def insert_traffic_flow(conn, flow_data):
    """Insert traffic flow data into normalized database schema"""
    c = conn.cursor()
    inserted = 0
    current_timestamp = int(datetime.now().timestamp())
    
    if not flow_data or 'flowSegmentData' not in flow_data:
        return inserted
    
    # Handle single segment or multiple segments
    segment_data = flow_data['flowSegmentData']
    segments = [segment_data] if isinstance(segment_data, dict) else segment_data
    
    for segment in segments:
        coordinates = segment.get('coordinates', {})
        coord_list = coordinates.get('coordinate', []) if isinstance(coordinates, dict) else coordinates
        
        if not coord_list:
            continue
            
        first_coord = coord_list[0]
        
        # Create stable segment_id from coordinates and frc
        segment_id_string = f"{json.dumps(coord_list, sort_keys=True)}_{segment.get('frc', '')}"
        segment_id = hashlib.sha256(segment_id_string.encode()).hexdigest()[:16]
        hash_id = f"{segment_id}_{current_timestamp}"
        
        # Insert road segment (stable data)
        road_segment_record = (
            segment_id,
            segment.get('openLr', ''),
            str(segment.get('frc', '')),
            json.dumps(coord_list),
            float(first_coord.get('latitude', 0)),
            float(first_coord.get('longitude', 0)),
            current_timestamp
        )
        
        c.execute('''
            INSERT OR IGNORE INTO road_segments (
                segment_id, openlr, frc, coordinates, coordinate_lat, coordinate_lon, created_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', road_segment_record)
        
        # Check for duplicate traffic data
        existing = c.execute('SELECT COUNT(*) FROM traffic_data WHERE hash_id = ?', (hash_id,)).fetchone()[0]
        if existing > 0:
            continue
        
        # Insert traffic data (time-varying data)
        traffic_data_record = (
            segment_id,
            hash_id,
            current_timestamp,
            float(segment.get('currentSpeed', 0)) if segment.get('currentSpeed') else None,
            float(segment.get('freeFlowSpeed', 0)) if segment.get('freeFlowSpeed') else None,
            int(segment.get('currentTravelTime', 0)) if segment.get('currentTravelTime') else None,
            int(segment.get('freeFlowTravelTime', 0)) if segment.get('freeFlowTravelTime') else None,
            float(segment.get('confidence', 0)) if segment.get('confidence') else None,
            1 if segment.get('roadClosure') else 0,
            str(segment.get('@version', ''))
        )
        
        c.execute('''
            INSERT INTO traffic_data (
                segment_id, hash_id, timestamp, currentSpeed, freeFlowSpeed, 
                currentTravelTime, freeFlowTravelTime, confidence, roadClosure, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', traffic_data_record)
        inserted += 1
    
    conn.commit()
    return inserted

def fetch_tomtom_traffic_flow_from_polylines(api_key, conn=None):
    """
    Fetch traffic flow data from sample points along Arizona road network polylines.
    Makes TomTom API calls and stores data in normalized database schema.
    """
    sample_points = load_geojson_polylines()
    
    if not sample_points:
        print("No sample points loaded from GeoJSON files")
        return 0
    
    total_pts = len(sample_points)
    batch_data = []
    total_inserted = 0
    base_url = 'https://api.tomtom.com'

    print(f"Starting TomTom API collection: {total_pts} points")
    print(f"Parameters: zoom={API_ZOOM}, thickness={API_THICKNESS}, batch={BATCH_SIZE}")

    for cnt, point_data in enumerate(sample_points, 1):
        lat, lon = point_data['lat'], point_data['lon']
        point = f"{lat},{lon}"
        
        url = f'{base_url}/traffic/services/4/flowSegmentData/absolute/{API_ZOOM}/json'
        params = {
            'key': api_key,
            'point': point,
            'unit': 'mph',
            'thickness': API_THICKNESS,
            'openLr': False
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'flowSegmentData' in data and data['flowSegmentData']:
                batch_data.append(data)
                if cnt % 10 == 0:  # Progress every 10 calls
                    print(f"Progress: {cnt}/{total_pts} ({cnt/total_pts*100:.1f}%)")
            
        except requests.exceptions.RequestException as e:
            if cnt % 10 == 0:  # Only log errors occasionally
                print(f"API error at point {cnt}: {e}")
            
        time.sleep(0.2)  # Rate limiting
        
        # Batch database insert
        if cnt % BATCH_SIZE == 0 and batch_data and conn:
            batch_inserted = sum(insert_traffic_flow(conn, data) for data in batch_data)
            total_inserted += batch_inserted
            print(f"Batch {cnt//BATCH_SIZE}: inserted {batch_inserted} records (total: {total_inserted})")
            batch_data = []
    
    # Final batch
    if batch_data and conn:
        batch_inserted = sum(insert_traffic_flow(conn, data) for data in batch_data)
        total_inserted += batch_inserted
    
    print(f"API collection completed: {total_inserted} traffic records inserted")
    return total_inserted

def save_sample_points():
    """Save sample points to JSON file for reference and testing"""
    sample_points = load_geojson_polylines()
    if sample_points:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tomtom_sample_points.json')
        try:
            with open(output_file, 'w') as f:
                json.dump(sample_points, f, indent=2)
            print(f"Sample points saved: {output_file} ({len(sample_points)} points)")
        except Exception as e:
            print(f"Error saving sample points: {e}")
    return len(sample_points) if sample_points else 0
def main():
    """Main function to run TomTom traffic data collection"""
    print("TomTom Traffic Data Collection - Arizona Road Networks")
    print("=" * 60)
    
    api_key = os.getenv('TOMTOM_API_KEY')
    
    if api_key:
        print("TomTom API key found - collecting traffic data")
        print(f"Parameters: zoom={API_ZOOM}, thickness={API_THICKNESS}, degree_step={DEGREE_STEP}, batch={BATCH_SIZE}")
        
        # Create database and clear existing traffic data
        conn = create_database()
        c = conn.cursor()
        c.execute("DELETE FROM traffic_data")
        conn.commit()
        print("Cleared existing traffic data")
        
        # Fetch and insert new data
        total_inserted = fetch_tomtom_traffic_flow_from_polylines(api_key, conn=conn)
        
        # Show summary
        road_segments_count = c.execute("SELECT COUNT(*) FROM road_segments").fetchone()[0]
        traffic_data_count = c.execute("SELECT COUNT(*) FROM traffic_data").fetchone()[0]
        
        print(f"\nCollection Summary:")
        print(f"  Road segments: {road_segments_count}")
        print(f"  Traffic records: {traffic_data_count}")
        print(f"  Database: tomtom.db")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        conn.close()
        
    else:
        print("TomTom API key not found - generating sample points only")
        print("Set TOMTOM_API_KEY environment variable to collect traffic data")
        
        # Generate and save sample points for testing
        total_points = save_sample_points()
        
        print(f"\nGenerated {total_points} sample points from Arizona road networks")
        print("Sample points saved to: tomtom_sample_points.json")
        print("\nDatabase Schema:")
        print("  - road_segments: stable segment info (segment_id, frc, coordinates)")
        print("  - traffic_data: time-varying traffic info (foreign key to road_segments)")

if __name__ == "__main__":
    main()
