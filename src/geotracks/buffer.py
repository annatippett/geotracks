import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import geopandas as gpd
from shapely.geometry import Point, LineString
import shapely.vectorized
import warnings

warnings.filterwarnings("ignore")


#  define all functions

# Extract point coordinates from the GeoDataFrame
def get_interpolated_coords(track_loc):
    
    interpolation_factor=10
    coords = np.array(list(zip(track_loc.lon.values, track_loc.lat.values)))
    
    # Calculate the number of segments
    num_segments = len(coords) - 1
    
    # Create an array to hold interpolated points
    interpolated_coords = np.zeros((num_segments * interpolation_factor, 2))
    
    for i in range(num_segments):
        start_coord = coords[i]
        end_coord = coords[i + 1]
    
        # Interpolate between the start and end points
        interpolation = np.linspace(0, 1, interpolation_factor, endpoint=False)
        segment_coords = (1 - interpolation[:, np.newaxis]) * start_coord + interpolation[:, np.newaxis] * end_coord
    
        # Store the interpolated points in the result array
        interpolated_coords[i * interpolation_factor:(i + 1) * interpolation_factor] = segment_coords

    return interpolated_coords


def get_buffer(ship_loc_df):
    points = [Point(xy) for xy in zip(ship_loc_df.lons, ship_loc_df.lats)]
    points = gpd.GeoDataFrame(ship_loc_df, geometry=points, crs="EPSG:4326")

    lines = points.groupby(["ID"])["geometry"].apply(lambda x: LineString(x.tolist()))
    lines = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326")
    lines.reset_index(inplace=True)

    # leave in degree coordinate system, so that longitudes are not wrapped around
    # lines = lines.to_crs("3857")

    # # Get left / right buffer (1 degree either side)
    buf_right = lines.buffer(distance=1, single_sided=True)
    buf_left = lines.buffer(distance=-1, single_sided=True)

    # WILL ALL BE ON ROTATED POLE
    buf_right = gpd.GeoSeries(buf_right, crs="4326")
    buf_left = gpd.GeoSeries(buf_left, crs="4326")

    return buf_right, buf_left, points

def buffer_data(polygon,lon,lat,data):
    poly_lon = np.where(polygon, lon, np.nan)
    poly_lat = np.where(polygon, lat, np.nan)
    poly_lon = poly_lon[np.isfinite(poly_lon)]
    poly_lat = poly_lat[np.isfinite(poly_lat)]

    poly_data = np.where(polygon, data, np.nan)
    poly_data = poly_data[np.isfinite(poly_data)]
    poly_data = np.where(poly_data < 0, np.nan, poly_data)

    return poly_lon, poly_lat, poly_data


def get_ref_grid_tree(points):
    ref_grid = list(zip(points.lats.values, points.lons.values))
    ref_grid = np.deg2rad(ref_grid) # convert to radians for haversine formula
    tree = BallTree(ref_grid, metric="haversine")
    return tree


def get_matching_points(lon,lat):
    matching_points = list(zip(lat, lon))
    matching_points = np.deg2rad(matching_points)
    return matching_points


def buf_data(points,lon,lat,data):
    tree = get_ref_grid_tree(points)
    
    matching_points = get_matching_points(lon,lat)
    
    distances, indices = tree.query(matching_points, k=1)
    # print(len(lon),len(lat),len(nd_buf),len(distances[:, 0]),len(indices[:, 0]))
    output = pd.DataFrame(
        {
            "lon" : lon,
            "lat" : lat,
            "track_index": indices[:, 0],
            "distance_from_track": distances[:, 0] * 6378, # converting from radians to distance in km by multiplying by radius of earth
            "data": data,
        }
    )  
    return output


def get_points_around_track(track_loc,data):
    # 1. interpolate track coords to avoid undercounting, put into dataframe
    track_loc=pd.DataFrame(track_loc,columns=["lon","lat","height"])
    interpolated_coords=get_interpolated_coords(track_loc)
    track_loc_lon = interpolated_coords[:,0]
    track_loc_lat = interpolated_coords[:,1]
    ship_loc_df = pd.DataFrame(
                    {"lons": track_loc_lon, "lats": track_loc_lat, "ID": 1}
                )
    
    # 2. get buffer around track
    buf_right, buf_left, points = get_buffer(ship_loc_df)

    # 3. get the lon/lat of the data field that are within the buffer
    try:
        lon, lat = np.meshgrid(data.grid_longitude.values, data.grid_latitude.values)
    except:
        lon, lat = data.Longitude.values, data.Latitude.values
    right_polygon = shapely.vectorized.contains(buf_right.item(), lon, lat)
    left_polygon = shapely.vectorized.contains(buf_left.item(), lon, lat)

    # 4. get the data that is within the buffers
    right_lon, right_lat, right_data = buffer_data(right_polygon,lon,lat,data)
    left_lon, left_lat, left_data = buffer_data(left_polygon,lon,lat,data)

    # 5. get the distance from the track / time along track / data for these points and output dataframe
    if len(left_lon)!=0:               
        left_output=buf_data(points,left_lon,left_lat,left_data)
        left_output["distance_from_track"] *= -1

    if len(right_lon)!=0:               
        right_output=buf_data(points,right_lon,right_lat,right_data)

    # 6. combine outputs and calculate time along track
    df = pd.concat([left_output, right_output])
    df["time_along_track"] = df.track_index / 60  # 60 because track points are every minute

    # #  7. bin data
    # df["distance_from_track"] = df["distance_from_track"] // 2 * 2
    # df["time_along_track"] = df["time_along_track"] // 1 * 1
    
    return df


