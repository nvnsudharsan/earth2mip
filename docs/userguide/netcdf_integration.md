# NetCDF Data Integration Guide

This guide explains how to integrate your ERA5 NetCDF data into the earth2mip framework for weather forecasting.

## Overview

The earth2mip framework now supports NetCDF data sources through the `NetCDFDataSource` class. This allows you to use your own ERA5 NetCDF files as initial conditions for weather forecasting models.

## Supported NetCDF Formats

### Standard ERA5 NetCDF Structure

Your NetCDF files should follow this structure:

```python
import xarray as xr

# Standard dimensions
ds = xr.Dataset({
    't2m': (['time', 'latitude', 'longitude'], data),      # 2m temperature
    'u10m': (['time', 'latitude', 'longitude'], data),     # 10m U wind
    'v10m': (['time', 'latitude', 'longitude'], data),     # 10m V wind
    'sp': (['time', 'latitude', 'longitude'], data),       # Surface pressure
    'msl': (['time', 'latitude', 'longitude'], data),      # Mean sea level pressure
    # ... other variables
}, coords={
    'time': times,           # datetime objects
    'latitude': lats,        # 90 to -90 (0.25° resolution)
    'longitude': lons,       # 0 to 359.75 (0.25° resolution)
})
```

### Required Dimensions

- **time**: Temporal dimension with datetime objects
- **latitude**: Latitude coordinates (typically 90° to -90°)
- **longitude**: Longitude coordinates (typically 0° to 359.75°)

### Supported Variables

Common ERA5 variables that work well with earth2mip:

| Variable | Description | Units |
|----------|-------------|-------|
| `t2m` | 2-meter temperature | K |
| `u10m` | 10-meter U wind | m/s |
| `v10m` | 10-meter V wind | m/s |
| `sp` | Surface pressure | Pa |
| `msl` | Mean sea level pressure | Pa |
| `t850` | Temperature at 850 hPa | K |
| `u850` | U wind at 850 hPa | m/s |
| `v850` | V wind at 850 hPa | m/s |
| `z850` | Geopotential at 850 hPa | m²/s² |
| `t500` | Temperature at 500 hPa | K |
| `u500` | U wind at 500 hPa | m/s |
| `v500` | V wind at 500 hPa | m/s |
| `z500` | Geopotential at 500 hPa | m²/s² |

## Usage Examples

### 1. Single NetCDF File

```python
from earth2mip.initial_conditions.netcdf import NetCDFDataSource

# Create data source from single file
data_source = NetCDFDataSource.from_single_file(
    "/path/to/your/era5_data.nc",
    channel_names=["t2m", "u10m", "v10m", "sp", "msl"]
)

# Get data for specific time
import datetime
time = datetime.datetime(2023, 1, 1, 12, 0)
data = data_source[time]  # Shape: (n_channels, n_lat, n_lon)
```

### 2. Directory of NetCDF Files

```python
# Create data source from directory
data_source = NetCDFDataSource.from_directory(
    "/path/to/era5/files/",
    pattern="*.nc",
    channel_names=["t2m", "u10m", "v10m"]
)
```

### 3. Integration with earth2mip

```python
from earth2mip import initial_conditions

# Use with earth2mip's get_data_source function
data_source = initial_conditions.get_data_source(
    channel_names=["t2m", "u10m", "v10m"],
    netcdf="/path/to/your/era5_data.nc",
    initial_condition_source=initial_conditions.schema.InitialConditionSource.netcdf
)
```

### 4. Custom Dimension Names

If your NetCDF files use different dimension names:

```python
data_source = NetCDFDataSource(
    "/path/to/your/data.nc",
    channel_names=["temperature", "wind_u"],
    time_dim="time",
    lat_dim="lat",      # instead of "latitude"
    lon_dim="lon"       # instead of "longitude"
)
```

## Configuration in Weather Events

You can also configure NetCDF data sources in weather event JSON files:

```json
{
    "name": "my_forecast",
    "start_time": "2023-01-01T12:00:00",
    "initial_condition_source": "netcdf",
    "netcdf": "/path/to/your/era5_data.nc"
}
```

## Data Requirements

### Grid Resolution

- **Recommended**: 0.25° × 0.25° (721 × 1440 grid points)
- **Supported**: Any regular lat-lon grid
- **Latitude**: Typically 90° to -90°
- **Longitude**: Typically 0° to 359.75°

### Time Format

- **Format**: datetime objects in the time dimension
- **Frequency**: Any temporal frequency (hourly, 6-hourly, daily, etc.)
- **Coverage**: Ensure data exists for your forecast start time

### Data Types

- **Recommended**: float32 or float64
- **Missing values**: Use NaN for missing data
- **Units**: Use standard meteorological units (K, Pa, m/s, etc.)

## Performance Considerations

### File Organization

For best performance with multiple time steps:

1. **Single file per time period**: Organize files by month/year
2. **Consistent naming**: Use predictable file naming patterns
3. **Local storage**: Keep files on fast local storage when possible

### Memory Usage

- **Large datasets**: Consider using Dask for out-of-core processing
- **Chunking**: NetCDF files should be properly chunked for efficient access
- **Compression**: Use NetCDF compression to reduce file sizes

## Troubleshooting

### Common Issues

1. **Time not found**: Ensure your forecast time exists in the NetCDF files
2. **Missing variables**: Check that all requested channel names exist in the data
3. **Dimension mismatch**: Verify dimension names match your NetCDF structure
4. **Grid mismatch**: Ensure your grid resolution is compatible with the model

### Error Messages

- `"No data found for time: ..."`: The requested time is not available in your NetCDF files
- `"Channels not found in NetCDF files: ..."`: Some requested variables don't exist in the data
- `"Path does not exist: ..."`: The NetCDF file or directory path is incorrect

### Debugging Tips

```python
# Check what's in your NetCDF file
import xarray as xr
ds = xr.open_dataset("/path/to/your/data.nc")
print("Dimensions:", ds.dims)
print("Data variables:", list(ds.data_vars.keys()))
print("Coordinates:", list(ds.coords.keys()))
print("Time range:", ds.time.min().values, "to", ds.time.max().values)
```

## Advanced Usage

### Custom Data Processing

You can extend the `NetCDFDataSource` class for custom data processing:

```python
class CustomNetCDFDataSource(NetCDFDataSource):
    def __getitem__(self, time):
        # Get raw data
        data = super().__getitem__(time)
        
        # Apply custom processing
        data = self.apply_custom_processing(data)
        
        return data
    
    def apply_custom_processing(self, data):
        # Your custom processing here
        return data
```

### Integration with Other Data Sources

You can combine NetCDF data with other earth2mip data sources:

```python
# Use NetCDF for some variables, CDS for others
netcdf_source = NetCDFDataSource("/path/to/data.nc", ["t2m", "u10m"])
cds_source = initial_conditions.cds.DataSource(["v10m", "sp"])

# Combine data sources as needed
```

## Examples

See the complete examples in:
- `examples/05_netcdf_data_integration.py` - Comprehensive usage examples
- `test/test_netcdf_integration.py` - Unit tests and validation

## Support

For questions or issues with NetCDF integration:
1. Check the troubleshooting section above
2. Review the example code
3. Open an issue on the earth2mip GitHub repository
