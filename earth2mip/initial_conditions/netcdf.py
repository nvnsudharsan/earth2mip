# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NetCDF Data Source for ERA5 data"""

import datetime
import glob
import os
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from earth2mip import grid
from earth2mip.initial_conditions import base


class NetCDFDataSource(base.DataSource):
    """NetCDF Data Source for ERA5 data
    
    This class provides a data source interface for ERA5 NetCDF files.
    It supports both single files and directories of NetCDF files.
    
    Example usage:
        # Single file
        data_source = NetCDFDataSource("/path/to/era5_data.nc", channel_names=["t2m", "u10m"])
        
        # Directory of files
        data_source = NetCDFDataSource("/path/to/era5_files/", channel_names=["t2m", "u10m"])
    """
    
    def __init__(
        self, 
        path: Union[str, List[str]], 
        channel_names: Optional[List[str]] = None,
        time_dim: str = "time",
        lat_dim: str = "latitude", 
        lon_dim: str = "longitude",
        data_vars: Optional[List[str]] = None
    ):
        """
        Initialize NetCDF data source
        
        Args:
            path: Path to NetCDF file(s) or directory containing NetCDF files
            channel_names: List of channel names to extract. If None, uses all available variables
            time_dim: Name of time dimension in NetCDF files
            lat_dim: Name of latitude dimension in NetCDF files  
            lon_dim: Name of longitude dimension in NetCDF files
            data_vars: List of data variables to use as channels. If None, uses all data variables
        """
        self.path = path
        self.time_dim = time_dim
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        
        # Determine if path is file, directory, or list of files
        if isinstance(path, str):
            if os.path.isfile(path):
                self.file_paths = [path]
            elif os.path.isdir(path):
                # Find all NetCDF files in directory
                self.file_paths = glob.glob(os.path.join(path, "*.nc"))
                if not self.file_paths:
                    raise ValueError(f"No NetCDF files found in directory: {path}")
            else:
                raise ValueError(f"Path does not exist: {path}")
        else:
            self.file_paths = path
            
        # Load metadata from first file
        self._load_metadata()
        
        # Set channel names
        if channel_names is None:
            if data_vars is None:
                # Use all data variables as channels
                self._channel_names = list(self.metadata.data_vars.keys())
            else:
                self._channel_names = data_vars
        else:
            self._channel_names = channel_names
            
        # Validate channel names
        available_vars = list(self.metadata.data_vars.keys())
        missing_channels = [c for c in self._channel_names if c not in available_vars]
        if missing_channels:
            raise ValueError(f"Channels not found in NetCDF files: {missing_channels}")
    
    def _load_metadata(self):
        """Load metadata from the first NetCDF file"""
        with xr.open_dataset(self.file_paths[0]) as ds:
            self.metadata = ds
            self._lat = ds[self.lat_dim].values
            self._lon = ds[self.lon_dim].values
            
    @property
    def grid(self) -> grid.LatLonGrid:
        """Get the grid information"""
        return grid.LatLonGrid(lat=self._lat, lon=self._lon)
    
    @property
    def channel_names(self) -> List[str]:
        """Get the list of channel names"""
        return self._channel_names
    
    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        """
        Get data for a specific time
        
        Args:
            time: The time to extract data for
            
        Returns:
            Array of shape (n_channels, n_lat, n_lon)
        """
        # Find the file containing this time
        target_file = self._find_file_for_time(time)
        
        if target_file is None:
            raise ValueError(f"No data found for time: {time}")
            
        # Load data from the file
        with xr.open_dataset(target_file) as ds:
            # Select the time and channels
            try:
                # Try to select the exact time
                data = ds.sel({self.time_dim: time})
            except KeyError:
                # If exact time not found, try to find closest time
                try:
                    data = ds.sel({self.time_dim: time}, method="nearest")
                except KeyError:
                    raise ValueError(f"Time {time} not found in {target_file}")
            
            # Extract the requested channels
            channel_data = []
            for channel in self._channel_names:
                if channel in data.data_vars:
                    channel_data.append(data[channel].values)
                else:
                    raise ValueError(f"Channel {channel} not found in data")
            
            # Stack channels into array of shape (n_channels, n_lat, n_lon)
            return np.stack(channel_data, axis=0)
    
    def _find_file_for_time(self, time: datetime.datetime) -> Optional[str]:
        """Find which file contains data for the given time"""
        for file_path in self.file_paths:
            try:
                with xr.open_dataset(file_path) as ds:
                    if self.time_dim in ds.dims:
                        time_values = ds[self.time_dim].values
                        if hasattr(time_values, 'astype'):
                            # Convert to datetime if needed
                            if time_values.dtype.kind in ['M', 'm']:  # datetime64
                                time_values = time_values.astype('datetime64[ns]')
                            elif time_values.dtype.kind == 'f':  # numeric time
                                # Assume this is hours since some epoch
                                epoch = datetime.datetime(1900, 1, 1)
                                time_values = [epoch + datetime.timedelta(hours=float(t)) for t in time_values]
                            
                            # Check if time is in this file
                            if hasattr(time_values, '__contains__'):
                                if time in time_values:
                                    return file_path
                            else:
                                # For numeric times, check if time is within range
                                if hasattr(time_values, 'min') and hasattr(time_values, 'max'):
                                    min_time = time_values.min()
                                    max_time = time_values.max()
                                    if min_time <= time <= max_time:
                                        return file_path
            except Exception:
                # Skip files that can't be opened or don't have time dimension
                continue
                
        return None
    
    @classmethod
    def from_directory(
        cls, 
        directory: str, 
        pattern: str = "*.nc",
        **kwargs
    ) -> "NetCDFDataSource":
        """
        Create NetCDF data source from directory of files
        
        Args:
            directory: Directory containing NetCDF files
            pattern: Glob pattern to match files (default: "*.nc")
            **kwargs: Additional arguments passed to NetCDFDataSource
            
        Returns:
            NetCDFDataSource instance
        """
        file_paths = glob.glob(os.path.join(directory, pattern))
        if not file_paths:
            raise ValueError(f"No files matching pattern '{pattern}' found in {directory}")
        
        return cls(file_paths, **kwargs)
    
    @classmethod
    def from_single_file(
        cls, 
        file_path: str, 
        **kwargs
    ) -> "NetCDFDataSource":
        """
        Create NetCDF data source from single file
        
        Args:
            file_path: Path to NetCDF file
            **kwargs: Additional arguments passed to NetCDFDataSource
            
        Returns:
            NetCDFDataSource instance
        """
        return cls(file_path, **kwargs)
