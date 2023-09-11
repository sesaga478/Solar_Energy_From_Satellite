import rasterio
from rasterio.transform import from_origin
def georeference(big_tif,ori_tif,geo_tif):
  # Abrir la imagen de referencia para obtener su transformación y sistema de coordenadas
  with rasterio.open(ori_tif) as src_referencia:
      transform_referencia = src_referencia.transform
      crs_referencia = src_referencia.crs

  # Abrir la imagen a georreferenciar para obtener sus dimensiones
  with rasterio.open(big_tif) as src_a_georreferenciar:
      width, height = src_a_georreferenciar.width, src_a_georreferenciar.height

      # Crear una nueva imagen georreferenciada
      with rasterio.open(geo_tif, 'w',
                        driver='GTiff', width=width, height=height,
                        count=src_a_georreferenciar.count,
                        dtype=src_a_georreferenciar.dtypes[0],
                        crs=crs_referencia, transform=transform_referencia) as dst_georreferenciada:
          for band in range(1, src_a_georreferenciar.count + 1):
              data = src_a_georreferenciar.read(band)
              dst_georreferenciada.write(data, band)

  print("Proceso de georreferenciación completado.")