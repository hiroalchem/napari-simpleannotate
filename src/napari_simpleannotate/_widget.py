import os
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFileDialog, QListWidget
import skimage.io


if TYPE_CHECKING:
    import napari


class BboxQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.initUI()
        self.initLayers()

    def initUI(self):
        # Create button for opening a file
        self.open_file_button = QPushButton("Open File", self)
        self.open_file_button.clicked.connect(self.openFile)

        # Create button for opening a directory
        self.open_dir_button = QPushButton("Open Directory", self)
        self.open_dir_button.clicked.connect(self.openDirectory)

        # Create button for saving the bounding box annotations
        self.save_button = QPushButton("Save Annotations", self)
        self.save_button.clicked.connect(self.saveAnnotations)

        # Create a list widget for displaying the list of opened files
        self.listWidget = QListWidget()
        self.listWidget.itemClicked.connect(self.open_image)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.open_file_button)
        layout.addWidget(self.open_dir_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.listWidget)
        self.setLayout(layout)

    def initLayers(self):
        """Initializes the image and shapes layers in the napari viewer."""
        self.viewer.add_image(np.zeros((10, 10)), name="image_layer")
        self.viewer.add_shapes(name="bbox_layer")

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "/")
        if fname[0]:
            self.listWidget.addItem(fname[0])

    def openDirectory(self):
        dname = QFileDialog.getExistingDirectory(self, "Open directory", "/")
        if dname:
            files = os.listdir(dname)
            image_files = sorted([f for f in files if f.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
            for image_file in image_files:
                self.listWidget.addItem(os.path.join(dname, image_file))

    def open_image(self, item):
        """Opens an image and updates the image layer in the napari viewer."""

        image = skimage.io.imread(item.text())
        image_layer = self.viewer.layers["image_layer"]
        image_layer.data = image
        image_layer.reset_contrast_limits()
        self.viewer.reset_view()

    def saveAnnotations(self):
        """Saves the bounding box annotations in the shapes layer in YOLO format."""

        # Get the current image file name and the corresponding annotation file name
        current_image_file = self.listWidget.currentItem().text()
        annotation_file = os.path.splitext(current_image_file)[0] + ".txt"

        shapes_layer = self.viewer.layers["bbox_layer"]
        image_layer = self.viewer.layers["image_layer"]
        image_height, image_width = image_layer.data.shape
        shapes_data = shapes_layer.data

        annotations = []

        # For each shape (rectangle)
        for shape_data in shapes_data:
            # Calculate the center, width, and height of the shape
            y_min, x_min = map(int, shape_data[0])
            y_max, x_max = map(int, shape_data[2])

            # Clip the coordinates to the image boundaries
            y_min = np.clip(y_min, 0, image_height - 1)
            y_max = np.clip(y_max, 0, image_height - 1)
            x_min = np.clip(x_min, 0, image_width - 1)
            x_max = np.clip(x_max, 0, image_width - 1)

            x_center = ((x_max + x_min) / 2) / image_width
            y_center = ((y_max + y_min) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Append the annotation to the list
            # TODO: Add support for multiple classes
            annotations.append(f"0 {x_center} {y_center} {width} {height}")

        # Join all the annotations into a string
        annotations_str = "\n".join(annotations)

        # Save the annotations to a file
        with open(annotation_file, "w") as f:
            f.write(annotations_str)
