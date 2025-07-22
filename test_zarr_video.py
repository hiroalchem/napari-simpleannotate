#!/usr/bin/env python
"""Simple test script to verify zarr video conversion functionality."""

import napari
from napari_simpleannotate._bbox_video_widget import BboxVideoQWidget

def test_zarr_video_widget():
    # Create napari viewer
    viewer = napari.Viewer()
    
    # Create widget
    widget = BboxVideoQWidget(viewer)
    
    # Add widget to viewer
    viewer.window.add_dock_widget(widget, area='right')
    
    print("Widget created successfully!")
    print("You can now:")
    print("1. Click 'Open Video' to load a video file")
    print("2. Check/uncheck 'Use Zarr for faster loading' to enable/disable zarr conversion")
    print("3. The first time a video is loaded with zarr enabled, it will be converted")
    print("4. Subsequent loads will use the cached zarr file for faster loading")
    
    # Start napari
    napari.run()

if __name__ == "__main__":
    test_zarr_video_widget()