# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a napari plugin for simple image annotation that provides three main annotation workflows:

1. **Bounding Box Annotation (YOLO format)**: For object detection training data on images
2. **Video Bounding Box Annotation**: For object detection training data on video files
3. **Image Classification Labeling**: For image classification training data

The plugin provides GUI widgets that integrate with napari's image viewer to enable:

- Opening single images or directories of images
- Class management with automatic numbering (bbox) or simple text labels (classification)
- Bounding box annotation using napari's rectangle tool
- Image classification labeling with CSV export
- Saving annotations in YOLO format with class.yaml files or CSV format for classification

## Key Commands

### Testing
```bash
# Run tests with pytest
pytest -v --color=yes --cov=napari_simpleannotate --cov-report=xml

# Run tests with tox (all environments)
tox

# Run tests for specific Python version
tox -e py38-linux
```

### Linting and Code Quality
```bash
# Run ruff linting (configured in pyproject.toml)
ruff check src/

# Run ruff with auto-fix
ruff check --fix src/

# Run black formatting (line length: 120)
black src/

# Run flake8 (configured in setup.cfg)
flake8 src/
```

### Building and Installation
```bash
# Install in development mode
pip install -e .

# Install with testing dependencies
pip install -e .[testing]

# Build distribution
python -m build
```

## Architecture

### Core Components

- **BboxQWidget** (`_bbox_widget.py`): Widget class for bounding box annotation in YOLO format on images
- **BboxVideoQWidget** (`_bbox_video_widget.py`): Widget class for bounding box annotation on video files
- **LabelImgQWidget** (`_labelimg_widget.py`): Widget class for image classification labeling with CSV export
- **Utilities** (`_utils.py`): Helper functions for coordinate conversion, file I/O, and class management

### Widget Structure

All widgets follow a standard Qt widget pattern:
- `initUI()`: Sets up the user interface components
- `initVariables()`: Initializes internal state variables
- `initLayers()`: Configures napari layers for annotation

#### BboxQWidget Architecture
- Focuses on spatial annotation (bounding boxes) for images
- Integrates with napari's shapes layer for rectangle drawing
- Saves annotations in YOLO format with class numbering
- Supports single file or directory-based workflows

#### BboxVideoQWidget Architecture
- Focuses on temporal bounding box annotation for video files
- Integrates with napari-video plugin for video display
- Frame-aware annotation with temporal tracking
- Saves annotations with frame information in YOLO-extended format

#### LabelImgQWidget Architecture
- Focuses on whole-image classification
- Supports channel splitting for multi-channel images
- Manages labels via pandas DataFrame
- Auto-saves labels to CSV format

### Key Features

#### BboxQWidget Features
1. **File Management**: Supports opening single files or directories with automatic class.yaml detection
2. **Class Management**: Dynamic class creation with automatic ID assignment using `find_missing_number()`
3. **Coordinate System**: Uses YOLO format (center x, center y, width, height) with conversion utilities in `xywh2xyxy()`
4. **Integration**: Leverages napari's rectangle tool for annotation creation

#### BboxVideoQWidget Features
1. **Video Support**: Single video file loading with PyAV library integration
2. **Frame Management**: Current frame tracking and display with temporal navigation
3. **Video Formats**: Supports MP4, AVI, MOV, MKV, WMV, FLV, WebM formats via PyAV
4. **Extended YOLO Format**: Saves annotations as `class_id frame x_center y_center width height`
5. **Directory-based Storage**: Classes and annotations saved in video file directory
6. **Dependency Check**: Automatic PyAV library availability verification

#### LabelImgQWidget Features
1. **Directory Processing**: Recursive image discovery with multiple format support (PNG, TIF, JPG, JPEG, TIFF)
2. **Image Display**: Single image or channel-split display with contrast preservation
3. **Classification Workflow**: Simple click-to-label interface for rapid image classification
4. **Data Persistence**: Auto-save labels to CSV, class definitions to class.txt
5. **Resume Capability**: Automatically loads existing labels and classes from previous sessions

### Dependencies

- **napari**: Core image viewer and annotation platform
- **magicgui**: For BboxQWidget GUI creation
- **PyYAML**: For class.yaml file handling (BboxQWidget, BboxVideoQWidget)
- **QtPy**: Qt abstraction layer for cross-platform GUI
- **scikit-image**: Image I/O operations
- **pandas**: DataFrame management for classification labels (LabelImgQWidget)
- **pathlib**: Modern path handling for file discovery
- **av (PyAV)**: Video file support for BboxVideoQWidget

## Development Notes

- Plugin provides three widgets registered in `napari.yaml`: "Bbox annotation", "Bbox video annotation", and "Label image classification"
- All widgets follow consistent Qt widget patterns for maintainability
- LabelImgQWidget migrated from legacy magicgui implementation to modern Qt widgets
- BboxVideoQWidget adapted from BboxQWidget with video-specific enhancements
- Tests use pytest-qt for Qt widget testing
- Code style enforced by ruff (line length: 79) and black (line length: 120)
- Supports Python 3.8-3.10 across Linux, macOS, and Windows

### File Formats
- **BboxQWidget**: Outputs YOLO format (.txt) + class.yaml
- **BboxVideoQWidget**: Outputs extended YOLO format with frame info (.txt) + class.yaml
- **LabelImgQWidget**: Outputs labels.csv + class.txt
- Image widgets support PNG, TIF, JPG, JPEG, TIFF formats
- Video widget supports MP4, AVI, MOV, MKV, WMV, FLV, WebM formats

### Video Annotation Workflow
1. Install PyAV: `pip install av`
2. Load video file through "Bbox video annotation" widget
3. Navigate frames using napari's time slider
4. Create bounding boxes with class assignment
5. Annotations automatically include current frame number
6. Save creates `video_name.txt` and `class.yaml` in video directory

## Memories

- to memorize