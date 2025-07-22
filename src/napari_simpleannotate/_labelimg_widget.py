import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import skimage.io
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


class LabelImgQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.initUI()
        self.initVariables()
        self.initLayers()

    def initUI(self):
        # Create button for opening a directory
        self.open_dir_button = QPushButton("Open Directory", self)
        self.open_dir_button.clicked.connect(self.openDirectory)

        # Create checkbox for split channels
        self.split_channels_checkbox = QCheckBox("Split Channels", self)

        # Create a list widget for displaying the list of opened files
        self.file_list_label = QLabel("File List:")
        self.listWidget = QListWidget()
        self.listWidget.currentItemChanged.connect(self.load_image)

        # Create class management UI
        self.class_list_label = QLabel("Class List:")
        self.classlistWidget = QListWidget()
        self.classlistWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.classlistWidget.currentItemChanged.connect(self.class_selected)

        # Create text input for adding classes
        self.add_class_label = QLabel("Add/Delete Class:")
        self.class_textbox = QLineEdit()
        self.class_textbox.returnPressed.connect(self.add_class)

        # Create button for clearing the file list
        self.clear_button = QPushButton("Clear File List", self)
        self.clear_button.clicked.connect(self.clear_list)

        # Create button for saving labels
        self.save_button = QPushButton("Save Labels", self)
        self.save_button.clicked.connect(self.save_labels)

        # Layout setup
        layout = QVBoxLayout()

        # File operations section
        layout.addWidget(self.open_dir_button)
        layout.addWidget(self.split_channels_checkbox)
        layout.addWidget(self.file_list_label)
        layout.addWidget(self.listWidget)
        layout.addWidget(self.clear_button)

        # Class management section
        layout.addWidget(self.class_list_label)
        layout.addWidget(self.classlistWidget)
        layout.addWidget(self.add_class_label)
        layout.addWidget(self.class_textbox)

        # Save section
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def initVariables(self):
        self.target_dir = ""
        self.image_lists = []
        self.classlist = []
        self.df = pd.DataFrame()
        self.current_image_path = ""
        self.image_layer = None
        self.colors_dict = {}
        self.contrast_limits_dict = {}
        self.dtype_dict = {}

    def initLayers(self):
        """Initialize napari layers for image display."""

    def openDirectory(self):
        """Open directory and load image files."""
        dname = QFileDialog.getExistingDirectory(self, "Open directory", "/")
        if dname:
            self.target_dir = dname
            self.load_directory()

    def load_directory(self):
        """Load image files from the target directory."""
        if not self.target_dir:
            return

        # Get image files using pathlib with supported extensions
        target_path = Path(self.target_dir)
        png_files = list(target_path.glob("**/*.png"))
        tif_files = list(target_path.glob("**/*.tif"))
        jpg_files = list(target_path.glob("**/*.jpg"))
        jpeg_files = list(target_path.glob("**/*.jpeg"))
        tiff_files = list(target_path.glob("**/*.tiff"))

        all_files = png_files + tif_files + jpg_files + jpeg_files + tiff_files
        all_files = sorted(all_files)

        # Convert to relative paths
        self.image_lists = [str(x.relative_to(target_path)) for x in all_files]

        # Update file list widget
        self.listWidget.clear()
        for img_path in self.image_lists:
            self.listWidget.addItem(img_path)

        # Initialize DataFrame for labels
        self.df = pd.DataFrame({"label": [""] * len(self.image_lists)}, index=self.image_lists).astype(str)

        # Load previous CSV if exists
        try:
            csv_path = os.path.join(self.target_dir, "labels.csv")
            prev_df = pd.read_csv(csv_path, index_col=0).fillna("").astype(str)
            self.df.update(prev_df)
        except:
            pass

        # Load class file if exists
        self.load_class_file()

    def load_class_file(self):
        """Load class definitions from class.txt file."""
        try:
            class_file_path = os.path.join(self.target_dir, "class.txt")
            with open(class_file_path) as f:
                lines = f.read().splitlines()
                for line in lines:
                    if line and line not in self.classlist:
                        self.classlist.append(line)

            # Add classes from existing labels in DataFrame
            class_from_df = list(set(self.df["label"]))
            for c in class_from_df:
                if c and c not in self.classlist:
                    self.classlist.append(c)

            # Update class list widget
            self.update_class_list_widget()
        except:
            pass

    def update_class_list_widget(self):
        """Update the class list widget with current classes."""
        self.classlistWidget.clear()
        for class_name in self.classlist:
            self.classlistWidget.addItem(class_name)

    def clear_list(self):
        """Clear the file list."""
        self.listWidget.clear()
        self.image_lists = []
        self.df = pd.DataFrame()
        self.target_dir = ""

    def load_image(self, current_item, previous_item):
        """Load and display the selected image."""
        if not current_item:
            return

        image_path = current_item.text()
        self.current_image_path = image_path
        full_path = os.path.join(self.target_dir, image_path)

        try:
            img = skimage.io.imread(full_path)
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return

        # Store current layer properties before removing layers
        self.store_layer_properties()

        # Remove existing image layers
        self.remove_existing_layers()

        # Load image based on channel splitting preference
        if self.split_channels_checkbox.isChecked() and len(img.shape) == 3:
            self.load_split_channels(img)
        else:
            self.load_single_image(img)

        # Load and display current label for this image
        self.load_current_label()

    def store_layer_properties(self):
        """Store properties of current layers for restoration."""
        for layer in self.viewer.layers:
            if "l4c_images" in layer.name:
                if "ch" in layer.name:
                    channel = layer.name.split("ch")[1]
                    self.colors_dict[channel] = layer.colormap.name
                    self.contrast_limits_dict[channel] = layer.contrast_limits
                    self.dtype_dict[channel] = layer.data.dtype

    def remove_existing_layers(self):
        """Remove existing image layers."""
        layer_names = []
        for layer in self.viewer.layers:
            if "l4c_images" in layer.name:
                layer_names.append(layer.name)

        for name in layer_names:
            del self.viewer.layers[name]

    def load_split_channels(self, img):
        """Load image with split channels."""
        if len(img.shape) != 3:
            return

        _, _, channels = img.shape
        for c in range(channels):
            channel_str = str(c)

            # Restore previous settings if available
            if channel_str in self.colors_dict:
                color = self.colors_dict[channel_str]
                if self.dtype_dict[channel_str] == img.dtype:
                    contrast_limits = self.contrast_limits_dict[channel_str]
                else:
                    contrast_limits = None
            else:
                color = "gray"
                contrast_limits = None

            # Set default contrast limits based on dtype
            if contrast_limits is None:
                if img.dtype == np.uint8:
                    contrast_limits = [0, 255]
                elif img.dtype == np.uint16:
                    contrast_limits = [0, 65535]
                else:
                    contrast_limits = None

            # Add channel layer
            self.image_layer = self.viewer.add_image(
                img[:, :, c],
                name=f"l4c_images_ch{c}",
                blending="additive",
                colormap=color,
                contrast_limits=contrast_limits,
            )

            # Apply stored contrast limits if available
            if channel_str in self.contrast_limits_dict and self.dtype_dict.get(channel_str) == img.dtype:
                self.image_layer.contrast_limits = self.contrast_limits_dict[channel_str]

    def load_single_image(self, img):
        """Load image as a single layer."""
        self.image_layer = self.viewer.add_image(img, name="l4c_images")

    def load_current_label(self):
        """Load and display the current label for the selected image."""
        if not self.current_image_path or self.df.empty:
            return

        current_label = self.df.at[self.current_image_path, "label"]

        # Update class selection to match current label
        if current_label in self.classlist:
            # Find and select the corresponding item in the class list
            for i in range(self.classlistWidget.count()):
                item = self.classlistWidget.item(i)
                if item.text() == current_label:
                    self.classlistWidget.setCurrentItem(item)
                    break
        else:
            # Clear selection if no matching class
            self.classlistWidget.clearSelection()

    def class_selected(self, current_item, previous_item):
        """Handle class selection and save label for current image."""
        if not current_item or not self.current_image_path:
            return

        selected_class = current_item.text()

        # Save the label for the current image
        if not self.df.empty:
            self.df.at[self.current_image_path, "label"] = selected_class
            # Auto-save the labels
            self.save_labels_to_csv()

    def add_class(self):
        """Add or remove class from the class list."""
        class_name = self.class_textbox.text().strip()
        if not class_name:
            return

        if class_name in self.classlist:
            # Remove existing class
            self.classlist.remove(class_name)
        else:
            # Add new class
            self.classlist.append(class_name)

        # Update the class list widget
        self.update_class_list_widget()

        # Clear the text box
        self.class_textbox.clear()

        # Save class file
        self.save_class_file()

    def save_class_file(self):
        """Save class definitions to class.txt file."""
        if not self.target_dir:
            return

        try:
            class_file_path = os.path.join(self.target_dir, "class.txt")
            with open(class_file_path, mode="w") as f:
                f.write("\n".join(self.classlist))
        except Exception as e:
            print(f"Error saving class file: {e}")

    def save_labels(self):
        """Save labels to CSV file."""
        self.save_labels_to_csv()

    def save_labels_to_csv(self):
        """Save the DataFrame with labels to CSV file."""
        if not self.target_dir or self.df.empty:
            return

        try:
            csv_path = os.path.join(self.target_dir, "labels.csv")
            self.df.to_csv(csv_path)
            print(f"Labels saved to {csv_path}")
        except Exception as e:
            print(f"Error saving labels: {e}")
