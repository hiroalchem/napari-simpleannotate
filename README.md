# napari-simpleannotate

[![License BSD-3](https://img.shields.io/pypi/l/napari-simpleannotate.svg?color=green)](https://github.com/hiroalchem/napari-simpleannotate/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-simpleannotate.svg?color=green)](https://pypi.org/project/napari-simpleannotate)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-simpleannotate.svg?color=green)](https://python.org)
[![tests](https://github.com/hiroalchem/napari-simpleannotate/workflows/tests/badge.svg)](https://github.com/hiroalchem/napari-simpleannotate/actions)
[![codecov](https://codecov.io/gh/hiroalchem/napari-simpleannotate/branch/main/graph/badge.svg)](https://codecov.io/gh/hiroalchem/napari-simpleannotate)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-simpleannotate)](https://napari-hub.org/plugins/napari-simpleannotate)

A napari plugin for simple image and video annotation that provides three main annotation workflows:

1. **Bounding Box Annotation (YOLO format)**: For object detection training data on images
2. **Video Bounding Box Annotation**: For object detection training data on video files
3. **Image Classification Labeling**: For image classification training data

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

![overview](https://github.com/hiroalchem/napari-simpleannotate/raw/main/images/dog_and_cat.jpg)


## Installation

You can install `napari-simpleannotate` via [pip]:

    pip install napari-simpleannotate



To install latest development version :

    pip install git+https://github.com/hiroalchem/napari-simpleannotate.git


## How to use

### Getting Started

After installing napari-simpleannotate, launch napari and navigate to `Plugins > Add dock widget` to find three annotation widgets:

- **Bbox annotation**: For bounding box annotation on images
- **Bbox video annotation**: For bounding box annotation on video files  
- **Label image classification**: For image classification labeling

### Bounding Box Annotation (Images)

**Prerequisites**: None required

1. **Opening Files**:
   - Single file: Click `Open File` to select an image file
   - Directory: Click `Open Directory` to select a folder containing images
   - If a `class.yaml` file exists in the directory, you'll be prompted to load existing classes

2. **Class Management**:
   - Enter class names in the text box and click `Add class`
   - Classes are automatically assigned sequential IDs (0, 1, 2, ...)
   - Select a class and click `Delete selected class` to remove it
   - Classes are saved to `class.yaml` alongside annotations

3. **Creating Annotations**:
   - Select a class from the list (becomes your active class)
   - Use napari's rectangle tool (shortcut: R) to draw bounding boxes
   - New rectangles automatically inherit the selected class
   - Change existing rectangles: select the shape, then click a different class

4. **Saving Work**:
   - Click `Save Annotations` to export in YOLO format
   - Files saved: `image_name.txt` (YOLO coordinates) + `class.yaml` (class definitions)
   - YOLO format: `class_id x_center y_center width height` (normalized 0-1)

### Video Bounding Box Annotation

**Prerequisites**: Install PyAV for video support: `pip install av`

1. **Opening Videos**:
   - Click `Open Video` to select MP4, AVI, MOV, MKV, WMV, FLV, or WebM files
   - Frames load through a cached reader that keeps recently viewed frames ready for scrubbing

2. **Navigation & Frame Tools**:
   - Use napari's time slider; the widget shows `Frame: current/total` alongside live cache usage
   - Jump between annotated frames with the `Prev (Q)` and `Next (W)` buttons or keyboard shortcuts
   - Nearby frames are prefetched in the background to keep playback smooth during review

3. **Annotating Frames**:
   - Draw rectangles after selecting a class; each shape stores its frame index automatically
   - Bounding boxes are only visible on the frame where they were created
   - Class management matches the image workflow, and `class.yaml` is written in YOLO `names:` format

4. **Automatic Tracking**:
   - Use `Start` to launch trackers for the boxes in the current frame and `Stop` to halt processing
   - Keep `Track all bounding boxes` enabled to follow every box, or disable it to track only selected shapes
   - Choose between the default OpenCV `CSRT` tracker and the experimental `TrackerVit`

   ![tracking demo](https://github.com/hiroalchem/napari-simpleannotate/raw/main/images/tracking_pigion_480w.gif)

5. **Saving & Exports**:
   - `Save Annotations` writes per-frame YOLO files (`imgNNN.txt`) in a video-specific folder next to the source
   - Annotated frames are saved once as `imgNNN.png` so you can review what was labeled later
   - Enable `Enable crop on save` to export fixed-size crops plus matching YOLO labels under `crops/`
   - Existing exports are reused when present so repeated saves only write new annotations

### Image Classification Labeling

**Prerequisites**: None required

1. **Opening Directory**:
   - Click `Open Directory` to select image folder
   - Recursively finds all images (PNG, TIF, JPG, JPEG, TIFF)
   - Automatically loads existing `labels.csv` and `class.txt` if present

2. **Display Options**:
   - **Split Channels**: Check to display multi-channel images as separate layers
   - Contrast settings preserved when switching between images
   - Navigate images using the file list on the left

3. **Labeling Workflow**:
   - Add classes: Type in text box and press Enter (or click `Add class`)
   - Remove classes: Type existing class name and press Enter
   - Assign labels: Select image → Click class name to label it
   - Real-time auto-save to `labels.csv` and `class.txt`

4. **Resume Sessions**:
   - Previous work automatically loaded when reopening directories
   - Continue labeling from where you left off

## Performance Notes

- **Video annotation**: Optimized with frame caching and parallel prefetching for smooth playback
- **Large datasets**: Classification widget handles thousands of images efficiently  
- **Memory management**: LRU cache prevents memory overflow during long annotation sessions

## Output Formats

| Widget | Annotation File | Class File | Format |
|--------|----------------|------------|---------|
| Bbox (Images) | `image.txt` | `class.yaml` | YOLO standard |
| Bbox (Video) | `imgNNN.txt` per frame (+ optional `crops/*.txt`) | `class.yaml` | YOLO normalized (per frame/crop) |
| Classification | `labels.csv` | `class.txt` | CSV with image-label pairs |


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-simpleannotate" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/hiroalchem/napari-simpleannotate/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
