#!/usr/bin/env python3
"""
Example: Using NetCDF ERA5 data with earth2mip

This example demonstrates how to integrate your own ERA5 NetCDF data
into the earth2mip framework for weather forecasting.

Requirements:
- ERA5 NetCDF files with standard dimensions (time, latitude, longitude)
- Variables like t2m, u10m, v10m, sp, msl, etc.
"""

import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from earth2mip import initial_conditions
from earth2mip.initial_conditions.netcdf import NetCDFDataSource


def create_sample_era5_netcdf(output_path: str):
    """Create a sample ERA5 NetCDF file for demonstration"""
    print("Creating sample ERA5 NetCDF file...")
    
    # Create sample data
    times = [
        datetime.datetime(2023, 1, 1, 0, 0),
        datetime.datetime(2023, 1, 1, 6, 0),
        datetime.datetime(2023, 1, 1, 12, 0),
        datetime.datetime(2023, 1, 1, 18, 0),
    ]
    
    # Standard ERA5 grid (0.25 degree resolution)
    lats = np.arange(90, -90.25, -0.25)  # 721 points
    lons = np.arange(0, 360, 0.25)       # 1440 points
    
    # Create sample variables
    data_vars = {}
    
    # 2-meter temperature (K)
    data_vars['t2m'] = (['time', 'latitude', 'longitude'], 
                       np.random.normal(280, 20, (len(times), len(lats), len(lons))))
    
    # 10-meter U wind (m/s)
    data_vars['u10m'] = (['time', 'latitude', 'longitude'], 
                        np.random.normal(0, 5, (len(times), len(lats), len(lons))))
    
    # 10-meter V wind (m/s)
    data_vars['v10m'] = (['time', 'latitude', 'longitude'], 
                        np.random.normal(0, 5, (len(times), len(lats), len(lons))))
    
    # Surface pressure (Pa)
    data_vars['sp'] = (['time', 'latitude', 'longitude'], 
                      np.random.normal(101325, 1000, (len(times), len(lats), len(lons))))
    
    # Mean sea level pressure (Pa)
    data_vars['msl'] = (['time', 'latitude', 'longitude'], 
                       np.random.normal(101325, 1000, (len(times), len(lats), len(lons))))
    
    # Create dataset
    ds = xr.Dataset(
        data_vars,
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        }
    )
    
    # Add attributes
    ds.attrs['title'] = 'Sample ERA5 data for earth2mip'
    ds.attrs['source'] = 'Generated for demonstration'
    
    # Save to NetCDF
    ds.to_netcdf(output_path)
    print(f"Sample NetCDF file created: {output_path}")
    
    return output_path


def example_1_single_netcdf_file():
    """Example 1: Using a single NetCDF file"""
    print("\n=== Example 1: Single NetCDF File ===")
    
    # Create sample data
    sample_file = "sample_era5_data.nc"
    create_sample_era5_netcdf(sample_file)
    
    try:
        # Create NetCDF data source
        data_source = NetCDFDataSource.from_single_file(
            sample_file,
            channel_names=["t2m", "u10m", "v10m", "sp", "msl"]
        )
        
        print(f"Grid shape: {data_source.grid.shape}")
        print(f"Channel names: {data_source.channel_names}")
        
        # Get data for a specific time
        time = datetime.datetime(2023, 1, 1, 12, 0)
        data = data_source[time]
        
        print(f"Data shape for {time}: {data.shape}")
        print(f"Expected shape: (n_channels, n_lat, n_lon) = (5, 721, 1440)")
        
        # Verify data shape
        assert data.shape == (5, 721, 1440), f"Unexpected data shape: {data.shape}"
        print("✓ Data shape is correct!")
        
    finally:
        # Clean up
        Path(sample_file).unlink(missing_ok=True)


def example_2_directory_of_netcdf_files():
    """Example 2: Using a directory of NetCDF files"""
    print("\n=== Example 2: Directory of NetCDF Files ===")
    
    # Create sample directory with multiple files
    data_dir = Path("sample_era5_data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Create multiple sample files
        for i, date in enumerate([datetime.datetime(2023, 1, 1), 
                                 datetime.datetime(2023, 1, 2)]):
            file_path = data_dir / f"era5_{date.strftime('%Y%m%d')}.nc"
            create_sample_era5_netcdf(str(file_path))
        
        # Create NetCDF data source from directory
        data_source = NetCDFDataSource.from_directory(
            str(data_dir),
            pattern="*.nc",
            channel_names=["t2m", "u10m", "v10m"]
        )
        
        print(f"Found {len(data_source.file_paths)} files")
        print(f"Channel names: {data_source.channel_names}")
        
        # Get data for different times
        for time in [datetime.datetime(2023, 1, 1, 12, 0),
                     datetime.datetime(2023, 1, 2, 12, 0)]:
            try:
                data = data_source[time]
                print(f"✓ Data retrieved for {time}: shape {data.shape}")
            except ValueError as e:
                print(f"✗ No data for {time}: {e}")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)


def example_3_integration_with_earth2mip():
    """Example 3: Integration with earth2mip's get_data_source function"""
    print("\n=== Example 3: Integration with earth2mip ===")
    
    # Create sample data
    sample_file = "sample_era5_data.nc"
    create_sample_era5_netcdf(sample_file)
    
    try:
        # Use earth2mip's get_data_source function
        data_source = initial_conditions.get_data_source(
            channel_names=["t2m", "u10m", "v10m"],
            netcdf=sample_file,
            initial_condition_source=initial_conditions.schema.InitialConditionSource.netcdf
        )
        
        print(f"Data source type: {type(data_source)}")
        print(f"Grid shape: {data_source.grid.shape}")
        print(f"Channel names: {data_source.channel_names}")
        
        # Get data for a specific time
        time = datetime.datetime(2023, 1, 1, 12, 0)
        data = data_source[time]
        
        print(f"✓ Successfully retrieved data for {time}: shape {data.shape}")
        
    finally:
        # Clean up
        Path(sample_file).unlink(missing_ok=True)


def example_4_custom_dimension_names():
    """Example 4: Handling custom dimension names in NetCDF files"""
    print("\n=== Example 4: Custom Dimension Names ===")
    
    # Create NetCDF with custom dimension names
    times = [datetime.datetime(2023, 1, 1, 0, 0)]
    lats = np.arange(90, -90.25, -0.25)
    lons = np.arange(0, 360, 0.25)
    
    ds = xr.Dataset(
        {
            'temperature': (['time', 'lat', 'lon'], 
                           np.random.normal(280, 20, (1, len(lats), len(lons)))),
            'wind_u': (['time', 'lat', 'lon'], 
                      np.random.normal(0, 5, (1, len(lats), len(lons)))),
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons,
        }
    )
    
    custom_file = "custom_dimensions.nc"
    ds.to_netcdf(custom_file)
    
    try:
        # Create data source with custom dimension names
        data_source = NetCDFDataSource(
            custom_file,
            channel_names=["temperature", "wind_u"],
            time_dim="time",
            lat_dim="lat",
            lon_dim="lon"
        )
        
        print(f"Custom dimension names handled successfully")
        print(f"Channel names: {data_source.channel_names}")
        
        # Get data
        time = datetime.datetime(2023, 1, 1, 0, 0)
        data = data_source[time]
        print(f"✓ Data retrieved: shape {data.shape}")
        
    finally:
        # Clean up
        Path(custom_file).unlink(missing_ok=True)


def main():
    """Run all examples"""
    print("ERA5 NetCDF Data Integration Examples")
    print("=" * 50)
    
    try:
        example_1_single_netcdf_file()
        example_2_directory_of_netcdf_files()
        example_3_integration_with_earth2mip()
        example_4_custom_dimension_names()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Replace sample data with your actual ERA5 NetCDF files")
        print("2. Adjust channel names to match your data variables")
        print("3. Use the data source in your earth2mip workflows")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
