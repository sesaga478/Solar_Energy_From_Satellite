import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

def raster_to_shapefile(input_raster, output_shapefile):
    # Read the raster
    with rasterio.open(input_raster) as src:
        # Read the raster data
        band = src.read(1)

        # Convert the raster to vector shapes
        vector_shapes = list(shapes(band, mask=None, transform=src.transform))

        # Create a geopandas GeoDataFrame from the shapes
        records = []
        for geom, value in vector_shapes:
            if value == 0:  # Modify this condition based on your road value
                continue
            records.append({'geometry': shape(geom)})

        gdf = gpd.GeoDataFrame(records)
        
        # Set the active geometry column
        gdf.set_geometry('geometry', inplace=True)

        # Assign the CRS to the GeoDataFrame
        gdf.crs = src.crs

        # Save the GeoDataFrame as a shapefile
        gdf.to_file(output_shapefile)