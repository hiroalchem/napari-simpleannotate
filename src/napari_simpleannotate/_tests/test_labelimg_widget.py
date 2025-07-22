import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from napari_simpleannotate import LabelImgQWidget

# Skip GUI-intensive tests to avoid segfaults - these tests are problematic with napari GUI initialization
pytestmark = pytest.mark.skip(reason="GUI tests cause segmentation faults with napari viewer initialization")


def test_labelimg_widget_init(make_napari_viewer):
    """Test LabelImgQWidget initialization."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    # Check that widget is properly initialized
    assert widget.viewer == viewer
    assert hasattr(widget, "classlistWidget")
    assert hasattr(widget, "class_textbox")
    assert hasattr(widget, "listWidget")
    assert hasattr(widget, "split_channels_checkbox")
    assert isinstance(widget.df, pd.DataFrame)
    assert widget.target_dir == ""


def test_add_class(make_napari_viewer):
    """Test adding a class to the class list."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    # Set up a temporary directory for class file saving
    with tempfile.TemporaryDirectory() as temp_dir:
        widget.target_dir = temp_dir

        # Add a class
        widget.class_textbox.setText("cat")
        widget.add_class()

        # Check that class was added
        assert "cat" in widget.classlist
        assert widget.classlistWidget.count() == 1
        assert widget.class_textbox.text() == ""  # Should be cleared


def test_remove_existing_class(make_napari_viewer):
    """Test removing an existing class from the class list."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    with tempfile.TemporaryDirectory() as temp_dir:
        widget.target_dir = temp_dir

        # Add a class
        widget.classlist = ["cat", "dog"]
        widget.update_class_list_widget()

        # Remove existing class
        widget.class_textbox.setText("cat")
        widget.add_class()

        # Check that class was removed
        assert "cat" not in widget.classlist
        assert "dog" in widget.classlist


@patch("qtpy.QtWidgets.QFileDialog.getExistingDirectory")
def test_open_directory(mock_dialog, make_napari_viewer):
    """Test opening a directory with images."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image files
        test_files = ["image1.png", "image2.jpg", "image3.tiff", "document.txt"]
        for filename in test_files:
            Path(temp_dir, filename).touch()

        mock_dialog.return_value = temp_dir
        widget.openDirectory()

        # Check that only image files were loaded
        assert len(widget.image_lists) == 3
        assert "document.txt" not in widget.image_lists
        assert widget.listWidget.count() == 3


def test_load_directory_with_existing_csv(make_napari_viewer):
    """Test loading directory with existing labels.csv file."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image files
        test_files = ["image1.png", "image2.jpg"]
        for filename in test_files:
            Path(temp_dir, filename).touch()

        # Create existing labels.csv
        existing_df = pd.DataFrame({"label": ["cat", "dog"]}, index=["image1.png", "image2.jpg"])
        existing_df.to_csv(os.path.join(temp_dir, "labels.csv"))

        widget.target_dir = temp_dir
        widget.load_directory()

        # Check that existing labels were loaded
        assert widget.df.at["image1.png", "label"] == "cat"
        assert widget.df.at["image2.jpg", "label"] == "dog"


def test_load_class_file(make_napari_viewer):
    """Test loading existing class.txt file."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create class.txt file
        class_file = os.path.join(temp_dir, "class.txt")
        with open(class_file, "w") as f:
            f.write("cat\ndog\nbird")

        widget.target_dir = temp_dir
        widget.load_class_file()

        # Check that classes were loaded to the internal list
        assert "cat" in widget.classlist
        assert "dog" in widget.classlist
        assert "bird" in widget.classlist
        # Check that the widget was updated
        assert widget.classlistWidget.count() == 3


def test_clear_list(make_napari_viewer):
    """Test clearing the file list."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    # Set up some data
    widget.image_lists = ["image1.png", "image2.jpg"]
    widget.listWidget.addItem("image1.png")
    widget.listWidget.addItem("image2.jpg")
    widget.df = pd.DataFrame({"label": ["cat", "dog"]})
    widget.target_dir = "/some/path"

    # Clear the list
    widget.clear_list()

    # Check that everything was cleared
    assert widget.listWidget.count() == 0
    assert len(widget.image_lists) == 0
    assert widget.df.empty
    assert widget.target_dir == ""


@patch("skimage.io.imread")
def test_load_image(mock_imread, make_napari_viewer):
    """Test loading and displaying an image."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up widget state
        widget.target_dir = temp_dir
        widget.image_lists = ["test.png"]
        widget.df = pd.DataFrame({"label": ["cat"]}, index=["test.png"])

        # Mock image data
        mock_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        # Create mock list item
        mock_item = MagicMock()
        mock_item.text.return_value = "test.png"

        # Load image
        widget.load_image(mock_item, None)

        # Check that image was loaded
        assert widget.current_image_path == "test.png"
        mock_imread.assert_called_once()


def test_class_selected(make_napari_viewer):
    """Test class selection and label saving."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up widget state
        widget.target_dir = temp_dir
        widget.current_image_path = "test.png"
        widget.df = pd.DataFrame({"label": [""]}, index=["test.png"])

        # Create mock list item
        mock_item = MagicMock()
        mock_item.text.return_value = "cat"

        with patch.object(widget, "save_labels_to_csv") as mock_save:
            widget.class_selected(mock_item, None)

        # Check that label was saved
        assert widget.df.at["test.png", "label"] == "cat"
        mock_save.assert_called_once()


def test_save_labels_to_csv(make_napari_viewer):
    """Test saving labels to CSV file."""
    viewer = make_napari_viewer()
    widget = LabelImgQWidget(viewer)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up widget state
        widget.target_dir = temp_dir
        widget.df = pd.DataFrame({"label": ["cat", "dog"]}, index=["image1.png", "image2.jpg"])

        # Save labels
        widget.save_labels_to_csv()

        # Check that CSV file was created and contains correct data
        csv_path = os.path.join(temp_dir, "labels.csv")
        assert os.path.exists(csv_path)

        loaded_df = pd.read_csv(csv_path, index_col=0)
        assert loaded_df.at["image1.png", "label"] == "cat"
        assert loaded_df.at["image2.jpg", "label"] == "dog"
