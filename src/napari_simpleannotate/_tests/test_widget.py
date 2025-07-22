import os
import pytest

from napari_simpleannotate import (
    BboxQWidget,
    BboxVideoQWidget,
    LabelImgQWidget,
)

# Skip GUI-intensive tests to avoid segfaults - these tests are problematic with napari GUI initialization
pytestmark = pytest.mark.skip(reason="GUI tests cause segmentation faults with napari viewer initialization")


def test_all_widgets_can_be_imported():
    """Test that all widgets can be imported without errors."""
    assert BboxQWidget is not None
    assert LabelImgQWidget is not None
    assert BboxVideoQWidget is not None


def test_all_widgets_can_be_instantiated(make_napari_viewer):
    """Test that all widgets can be instantiated with a napari viewer."""
    viewer = make_napari_viewer()

    # Test BboxQWidget
    bbox_widget = BboxQWidget(viewer)
    assert bbox_widget.viewer == viewer

    # Test LabelImgQWidget
    labelimg_widget = LabelImgQWidget(viewer)
    assert labelimg_widget.viewer == viewer

    # Test BboxVideoQWidget
    video_widget = BboxVideoQWidget(viewer)
    assert video_widget.viewer == viewer


def test_widgets_have_required_attributes(make_napari_viewer):
    """Test that all widgets have the required basic attributes."""
    viewer = make_napari_viewer()

    widgets = [BboxQWidget(viewer), LabelImgQWidget(viewer), BboxVideoQWidget(viewer)]

    for widget in widgets:
        # All widgets should have these basic attributes
        assert hasattr(widget, "viewer")
        assert hasattr(widget, "classlistWidget")
        assert hasattr(widget, "class_textbox")

        # All widgets should have these methods
        assert hasattr(widget, "initUI")
        assert hasattr(widget, "initVariables")
        assert hasattr(widget, "initLayers")
        assert callable(getattr(widget, "add_class", None))
