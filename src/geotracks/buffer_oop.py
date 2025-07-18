import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from sklearn.neighbors import BallTree
from rasterio.features import geometry_mask
from affine import Affine
import warnings

warnings.filterwarnings("ignore")


class BufferBase:
    def __init__(self, track_data):
        self.track_data = pd.DataFrame(track_data, columns=["lon", "lat", "height"])
        self.interpolated_coords = self._interpolate_coords()
        self.ship_df = pd.DataFrame(
            {
                "lons": self.interpolated_coords[:, 0],
                "lats": self.interpolated_coords[:, 1],
                "ID": 1,
            }
        ).dropna()
        self.buf_right, self.buf_left, self.points = self._get_buffer()

    def _interpolate_coords(self, factor=10):
        coords = self.track_data[["lon", "lat"]].values
        segments = len(coords) - 1
        interpolated = np.zeros((segments * factor, 2))

        for i in range(segments):
            start, end = coords[i], coords[i + 1]
            interp = np.linspace(0, 1, factor, endpoint=False)[:, np.newaxis]
            interpolated[i * factor : (i + 1) * factor] = (
                1 - interp
            ) * start + interp * end
        return interpolated

    def _get_buffer(self):
        points = [Point(xy) for xy in zip(self.ship_df.lons, self.ship_df.lats)]
        points = gpd.GeoDataFrame(self.ship_df, geometry=points, crs="EPSG:4326")
        lines = points.groupby("ID")["geometry"].apply(lambda x: LineString(x.tolist()))
        lines = gpd.GeoDataFrame(
            lines, geometry="geometry", crs="EPSG:4326"
        ).reset_index()
        buf_right = gpd.GeoSeries(
            lines.buffer(distance=1, single_sided=True), crs="EPSG:4326"
        )
        buf_left = gpd.GeoSeries(
            lines.buffer(distance=-1, single_sided=True), crs="EPSG:4326"
        )
        return buf_right, buf_left, points

    def _geometry_mask(self, polygon, lon, lat):
        transform = self._calculate_affine_transform(lon, lat)
        return geometry_mask(
            [polygon], transform=transform, invert=True, out_shape=lon.shape
        )

    def _calculate_affine_transform(self, lon, lat):
        lon_res = (lon[0, -1] - lon[0, 0]) / (lon.shape[1] - 1)
        lat_res = (lat[-1, 0] - lat[0, 0]) / (lat.shape[0] - 1)
        return Affine.translation(
            lon[0, 0] - lon_res / 2, lat[0, 0] - lat_res / 2
        ) * Affine.scale(lon_res, lat_res)

    def _buffer_data(self, mask, lon, lat, data):
        valid_mask = mask & (data >= 0)
        return lon[valid_mask], lat[valid_mask], data[valid_mask]

    def _query_points(self, lon, lat, data):
        ref_grid = np.deg2rad(
            list(zip(self.points.lats.values, self.points.lons.values))
        )
        tree = BallTree(ref_grid, metric="haversine")
        query_points = np.deg2rad(list(zip(lat, lon)))
        distances, indices = tree.query(query_points, k=1)

        return pd.DataFrame(
            {
                "lon": lon,
                "lat": lat,
                "track_index": indices[:, 0],
                "distance_from_track": distances[:, 0] * 6378,
                "data": data,
            }
        )

    def _getlonlat(self, data):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def extract(self, data):
        lon, lat = self._getlonlat(data)

        if len(self.buf_right) == 0 or len(self.buf_left) == 0:
            return pd.DataFrame(
                columns=[
                    "lon",
                    "lat",
                    "track_index",
                    "distance_from_track",
                    "data",
                    "time_along_track",
                ]
            )

        right_mask = self._geometry_mask(self.buf_right.unary_union, lon, lat)
        left_mask = self._geometry_mask(self.buf_left.unary_union, lon, lat)

        rlon, rlat, rdata = self._buffer_data(right_mask, lon, lat, data.values)
        llon, llat, ldata = self._buffer_data(left_mask, lon, lat, data.values)

        dfs = []
        if len(rdata) > 0:
            dfs.append(self._query_points(rlon, rlat, rdata))
        if len(ldata) > 0:
            left_df = self._query_points(llon, llat, ldata)
            left_df["distance_from_track"] *= -1
            dfs.append(left_df)

        if not dfs:
            return pd.DataFrame(
                columns=[
                    "lon",
                    "lat",
                    "track_index",
                    "distance_from_track",
                    "data",
                    "time_along_track",
                ]
            )

        df = pd.concat(dfs)
        df["time_along_track"] = df.track_index / 60
        return df


class ModelDataBuffer(BufferBase):
    def _getlonlat(self, data):
        lat_name = [dim for dim in data.dims if "latitude" in dim][0]
        lon_name = [dim for dim in data.dims if "longitude" in dim][0]
        lon, lat = np.meshgrid(data[lon_name].values, data[lat_name].values)
        return lon, lat


class MODISDataBuffer(BufferBase):
    def _getlonlat(self, data):
        lon, lat = data.Longitude.values, data.Latitude.values
        return lon, lat


class SEVIRIDataBuffer(BufferBase):
    def _getlonlat(self, data):
        lon, lat = np.meshgrid(data.Longitude.values, data.Latitude.values)
        return lon, lat
