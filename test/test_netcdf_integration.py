#!/usr/bin/env python3
"""
Tests for NetCDF data integration in earth2mip
"""

import datetime
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from earth2mip import initial_conditions
from earth2mip.initial_conditions.netcdf import NetCDFDataSource


class TestNetCDFDataSource:
    """Test cases for NetCDFDataSource class"""
    
    def create_sample_netcdf(self, file_path: str, times=None, custom_dims=None):
        """Create a sample NetCDF file for testing"""
        if times is None:
            times = [
                datetime.datetime(2023, 1, 1, 0, 0),
                datetime.datetime(2023, 1, 1, 6, 0),
                datetime.datetime(2023, 1, 1, 12, 0),
            ]
        
        # Standard ERA5-like grid
        lats = np.arange(90, -90.25, -0.25)  # 721 points
        lons = np.arange(0, 360, 0.25)       # 1440 points
        
        # Use custom dimension names if provided
        if custom_dims:
            time_dim = custom_dims.get('time', 'time')
            lat_dim = custom_dims.get('lat', 'latitude')
            lon_dim = custom_dims.get('lon', 'longitude')
        else:
            time_dim, lat_dim, lon_dim = 'time', 'latitude', 'longitude'
        
        # Create sample data
        data_vars = {
            't2m': ([time_dim, lat_dim, lon_dim], 
                   np.random.normal(280, 20, (len(times), len(lats), len(lons)))),
            'u10m': ([time_dim, lat_dim, lon_dim], 
                    np.random.normal(0, 5, (len(times), len(lats), len(lons)))),
            'v10m': ([time_dim, lat_dim, lon_dim], 
                    np.random.normal(0, 5, (len(times), len(lats), len(lons)))),
            'sp': ([time_dim, lat_dim, lon_dim], 
                  np.random.normal(101325, 1000, (len(times), len(lats), len(lons)))),
        }
        
        coords = {
            time_dim: times,
            lat_dim: lats,
            lon_dim: lons,
        }
        
        ds = xr.Dataset(data_vars, coords=coords)
        ds.to_netcdf(file_path)
        return file_path
    
    def test_single_file_creation(self):
        """Test creating NetCDFDataSource from single file"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            self.create_sample_netcdf(tmp.name)
            
            try:
                data_source = NetCDFDataSource.from_single_file(
                    tmp.name,
                    channel_names=["t2m", "u10m", "v10m"]
                )
                
                assert len(data_source.channel_names) == 3
                assert "t2m" in data_source.channel_names
                assert "u10m" in data_source.channel_names
                assert "v10m" in data_source.channel_names
                
                # Test grid properties
                assert data_source.grid.shape == (721, 1440)
                
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    
    def test_data_retrieval(self):
        """Test retrieving data for specific times"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            times = [
                datetime.datetime(2023, 1, 1, 0, 0),
                datetime.datetime(2023, 1, 1, 6, 0),
                datetime.datetime(2023, 1, 1, 12, 0),
            ]
            self.create_sample_netcdf(tmp.name, times=times)
            
            try:
                data_source = NetCDFDataSource.from_single_file(
                    tmp.name,
                    channel_names=["t2m", "u10m"]
                )
                
                # Test data retrieval for existing time
                time = datetime.datetime(2023, 1, 1, 6, 0)
                data = data_source[time]
                
                assert data.shape == (2, 721, 1440)  # 2 channels, 721 lats, 1440 lons
                assert isinstance(data, np.ndarray)
                
                # Test data retrieval for non-existing time
                with pytest.raises(ValueError, match="No data found for time"):
                    data_source[datetime.datetime(2023, 1, 2, 0, 0)]
                
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    
    def test_custom_dimension_names(self):
        """Test NetCDFDataSource with custom dimension names"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            custom_dims = {'time': 'time', 'lat': 'lat', 'lon': 'lon'}
            self.create_sample_netcdf(tmp.name, custom_dims=custom_dims)
            
            try:
                data_source = NetCDFDataSource(
                    tmp.name,
                    channel_names=["t2m", "u10m"],
                    time_dim="time",
                    lat_dim="lat",
                    lon_dim="lon"
                )
                
                # Test data retrieval
                time = datetime.datetime(2023, 1, 1, 0, 0)
                data = data_source[time]
                assert data.shape == (2, 721, 1440)
                
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    
    def test_invalid_channel_names(self):
        """Test error handling for invalid channel names"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            self.create_sample_netcdf(tmp.name)
            
            try:
                with pytest.raises(ValueError, match="Channels not found"):
                    NetCDFDataSource.from_single_file(
                        tmp.name,
                        channel_names=["t2m", "invalid_channel"]
                    )
                
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    
    def test_directory_creation(self):
        """Test creating NetCDFDataSource from directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple NetCDF files
            for i in range(3):
                file_path = Path(tmpdir) / f"era5_2023010{i+1}.nc"
                times = [datetime.datetime(2023, 1, i+1, 12, 0)]
                self.create_sample_netcdf(str(file_path), times=times)
            
            # Create data source from directory
            data_source = NetCDFDataSource.from_directory(
                tmpdir,
                pattern="*.nc",
                channel_names=["t2m", "u10m"]
            )
            
            assert len(data_source.file_paths) == 3
            assert len(data_source.channel_names) == 2
            
            # Test data retrieval from different files
            time1 = datetime.datetime(2023, 1, 1, 12, 0)
            time2 = datetime.datetime(2023, 1, 2, 12, 0)
            
            data1 = data_source[time1]
            data2 = data_source[time2]
            
            assert data1.shape == (2, 721, 1440)
            assert data2.shape == (2, 721, 1440)
    
    def test_integration_with_earth2mip(self):
        """Test integration with earth2mip's get_data_source function"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            self.create_sample_netcdf(tmp.name)
            
            try:
                # Test using get_data_source function
                data_source = initial_conditions.get_data_source(
                    channel_names=["t2m", "u10m"],
                    netcdf=tmp.name,
                    initial_condition_source=initial_conditions.schema.InitialConditionSource.netcdf
                )
                
                assert isinstance(data_source, NetCDFDataSource)
                assert len(data_source.channel_names) == 2
                
                # Test data retrieval
                time = datetime.datetime(2023, 1, 1, 0, 0)
                data = data_source[time]
                assert data.shape == (2, 721, 1440)
                
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    
    def test_missing_netcdf_path(self):
        """Test error handling when netcdf path is not provided"""
        with pytest.raises(ValueError, match="netcdf path must be provided"):
            initial_conditions.get_data_source(
                channel_names=["t2m", "u10m"],
                netcdf="",  # Empty path
                initial_condition_source=initial_conditions.schema.InitialConditionSource.netcdf
            )
    
    def test_nonexistent_file(self):
        """Test error handling for non-existent files"""
        with pytest.raises(ValueError, match="Path does not exist"):
            NetCDFDataSource.from_single_file("/nonexistent/file.nc")
    
    def test_empty_directory(self):
        """Test error handling for empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No files matching pattern"):
                NetCDFDataSource.from_directory(tmpdir, pattern="*.nc")


def test_netcdf_integration_example():
    """Integration test that runs the example code"""
    # This test ensures the example code works correctly
    try:
        from examples.05_netcdf_data_integration import (
            create_sample_era5_netcdf,
            example_1_single_netcdf_file
        )
        
        # Test the sample creation function
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            create_sample_era5_netcdf(tmp.name)
            
            # Verify the file was created and has correct structure
            with xr.open_dataset(tmp.name) as ds:
                assert 't2m' in ds.data_vars
                assert 'u10m' in ds.data_vars
                assert 'latitude' in ds.coords
                assert 'longitude' in ds.coords
                assert 'time' in ds.coords
                
            Path(tmp.name).unlink(missing_ok=True)
            
    except ImportError:
        # Skip if examples module is not available
        pytest.skip("Examples module not available")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
