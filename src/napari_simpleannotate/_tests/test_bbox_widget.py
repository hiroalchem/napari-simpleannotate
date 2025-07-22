import os
from unittest.mock import patch

import pytest

from napari_simpleannotate import BboxQWidget

# Skip GUI-intensive tests to avoid segfaults - these tests are problematic with napari GUI initialization
pytestmark = pytest.mark.skip(reason="GUI tests cause segmentation faults with napari viewer initialization")


def test_bbox_widget_init(make_napari_viewer):
    """Test BboxQWidget initialization."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Check that widget is properly initialized
    assert widget.viewer == viewer
    assert hasattr(widget, "classlistWidget")
    assert hasattr(widget, "class_textbox")
    assert hasattr(widget, "listWidget")
    assert hasattr(widget, "features")
    assert hasattr(widget, "text")


def test_add_class(make_napari_viewer):
    """Test adding a class to the class list."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Add a class
    widget.class_textbox.setText("person")
    widget.add_class()

    # Check that class was added
    assert widget.classlistWidget.count() == 1
    assert widget.classlistWidget.item(0).text() == "0: person"
    assert widget.class_textbox.text() == ""  # Should be cleared


def test_add_duplicate_class(make_napari_viewer):
    """Test that duplicate classes show a message."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Add a class first
    widget.class_textbox.setText("person")
    widget.add_class()

    # Try to add the same class again
    widget.class_textbox.setText("person")
    widget.add_class()

    # Should still have only one class (the actual behavior prints a message)
    # Note: In test mode, popup is skipped and append behavior is used
    assert widget.classlistWidget.count() == 1


def test_sort_classlist(make_napari_viewer):
    """Test that class list is properly sorted."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Add classes in non-sequential order
    widget.classlistWidget.addItem("2: car")
    widget.classlistWidget.addItem("0: person")
    widget.classlistWidget.addItem("1: bike")

    widget.sort_classlist()

    # Check order
    assert widget.classlistWidget.item(0).text() == "0: person"
    assert widget.classlistWidget.item(1).text() == "1: bike"
    assert widget.classlistWidget.item(2).text() == "2: car"


def test_del_class(make_napari_viewer):
    """Test deleting a class from the class list."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Add a class
    widget.class_textbox.setText("person")
    widget.add_class()

    # Select and delete it
    widget.classlistWidget.setCurrentRow(0)

    with patch("qtpy.QtWidgets.QMessageBox.question", return_value=16384):  # QMessageBox.Yes
        widget.del_class()

    # Check that class was deleted
    assert widget.classlistWidget.count() == 0


@patch("qtpy.QtWidgets.QFileDialog.getOpenFileName")
def test_open_file(mock_dialog, make_napari_viewer):
    """Test opening a single file."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Mock file dialog
    test_file = "/path/to/test.jpg"
    mock_dialog.return_value = (test_file, "")

    # Mock the open_image method completely to avoid file I/O
    with patch.object(widget, "open_image") as mock_open_image:
        widget.openFile()

    # Check that file was added to list
    assert widget.listWidget.count() == 1
    assert widget.listWidget.item(0).text() == test_file
    # Verify open_image was called with the list item
    assert mock_open_image.call_count == 1


@patch("qtpy.QtWidgets.QFileDialog.getExistingDirectory")
@patch("os.listdir")
def test_open_directory(mock_listdir, mock_dialog, make_napari_viewer):
    """Test opening a directory with images."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Mock directory dialog and file listing
    test_dir = "/path/to/images"
    mock_dialog.return_value = test_dir
    mock_listdir.return_value = ["image1.jpg", "image2.png", "document.txt", "image3.tiff"]

    widget.openDirectory()

    # Check that only image files were added
    assert widget.listWidget.count() == 3
    items = [widget.listWidget.item(i).text() for i in range(widget.listWidget.count())]
    expected_files = [
        os.path.join(test_dir, "image1.jpg"),
        os.path.join(test_dir, "image2.png"),
        os.path.join(test_dir, "image3.tiff"),
    ]
    assert sorted(items) == sorted(expected_files)


def test_clear_list(make_napari_viewer):
    """Test clearing the file list."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Add some files
    widget.listWidget.addItem("file1.jpg")
    widget.listWidget.addItem("file2.png")

    # Clear the list
    widget.listWidget.clear()

    # Check that list is empty
    assert widget.listWidget.count() == 0


def test_layers_initialization(make_napari_viewer):
    """Test that napari layers are properly initialized."""
    viewer = make_napari_viewer()
    widget = BboxQWidget(viewer)

    # Check that layers were added
    layer_names = [layer.name for layer in viewer.layers]
    assert "image_layer" in layer_names
    assert "bbox_layer" in layer_names

    # Check bbox layer properties
    bbox_layer = viewer.layers["bbox_layer"]
    # Check that features structure matches
    assert list(bbox_layer.features.columns) == list(widget.features.keys())
    assert bbox_layer.text == widget.text
