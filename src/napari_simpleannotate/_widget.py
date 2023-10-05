import os
from typing import TYPE_CHECKING

import numpy as np
import skimage.io
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget, QAbstractItemView

from ._utils import xywh2xyxy

if TYPE_CHECKING:
    import napari


class BboxQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.initUI()
        self.initVariables()
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

        # Create a list widget for displaying the list of classes
        self.classlistWidget = QListWidget()
        self.classlistWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.classlistWidget.itemClicked.connect(self.set_default_class)

        # Create text box for entering the class names
        self.class_textbox = QLineEdit()
        self.class_textbox.setPlaceholderText("Enter class name")

        # Create button for adding class to classlist
        self.add_class_button = QPushButton("Add class", self)
        self.add_class_button.clicked.connect(self.add_class)

        # Create button for deleting class from classlist
        self.del_class_button = QPushButton("Delete selected class", self)
        self.del_class_button.clicked.connect(self.del_class)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.open_file_button)
        layout.addWidget(self.open_dir_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.listWidget)
        layout.addWidget(self.classlistWidget)
        layout.addWidget(self.class_textbox)
        layout.addWidget(self.add_class_button)
        layout.addWidget(self.del_class_button)
        self.setLayout(layout)

    def initVariables(self):
        """Initializes the variables."""
        self.features = {"class": []}
        self.text = {
            "string": "{class}",
            "anchor": "upper_left",
            "translation": [-5, 0],
            "size": 8,
            "color": "green",
        }

    def initLayers(self):
        """Initializes the image and shapes layers in the napari viewer."""
        self.viewer.add_image(np.zeros((10, 10)), name="image_layer")
        self.viewer.add_shapes(name="bbox_layer", features=self.features, text=self.text)

    def set_default_class(self):
        """Sets the default class for the shapes layer."""
        shapes_layer = self.viewer.layers["bbox_layer"]
        selected_items = self.classlistWidget.selectedItems()
        if not selected_items:
            return
        if len(selected_items) > 1:
            print("Multiple classes selected")
            return
        for item in selected_items:
            shapes_layer.feature_defaults = {"class": item.text()}

    def add_class(self):
        """Adds the text in the class_textbox to the classlistWidget."""
        class_name = self.class_textbox.text()
        if class_name:
            if class_name in self.features["class"]:
                print("Class already exists")
                return
            self.classlistWidget.addItem(class_name)
            self.class_textbox.clear()
            self.features["class"].append(class_name)

    def del_class(self):
        """Deletes the selected class from the classlistWidget and the features dictionary."""
        selected_items = self.classlistWidget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            class_name = item.text()
            self.classlistWidget.takeItem(self.classlistWidget.row(item))
            self.features["class"].remove(class_name)

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "/")
        if fname[0]:
            self.listWidget.addItem(fname[0])
            item = self.listWidget.findItems(fname[0], Qt.MatchExactly)[0]
            self.listWidget.setCurrentItem(item)
            self.open_image(item)

    def openDirectory(self):
        dname = QFileDialog.getExistingDirectory(self, "Open directory", "/")
        if dname:
            files = os.listdir(dname)
            image_files = sorted([f for f in files if f.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
            for image_file in image_files:
                self.listWidget.addItem(os.path.join(dname, image_file))

    def open_image(self, item):
        """Opens an image and updates the image layer in the napari viewer."""
        image_file = item.text()
        image = skimage.io.imread(image_file)
        rgb = image.shape[-1] in (3, 4)
        if rgb:
            image_height, image_width, _ = image.shape[-3:]
        else:
            image_height, image_width = image.shape[-2:]
        image_layer = self.viewer.layers["image_layer"]
        image_layer.rgb = rgb
        image_layer.data = image
        image_layer.reset_contrast_limits()

        txt_file = os.path.splitext(image_file)[0] + ".txt"
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                lines = f.readlines()
                shapes_data = []
                for line in lines:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_min, y_min, x_max, y_max = xywh2xyxy(
                        [x_center, y_center, width, height], scale=(image_width, image_height)
                    )
                    shapes_data.append([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])
                shapes_layer = self.viewer.layers["bbox_layer"]
                shapes_layer.data = shapes_data
        else:
            shapes_layer = self.viewer.layers["bbox_layer"]
            shapes_layer.data = []

        self.viewer.reset_view()

    def saveAnnotations(self):
        """Saves the bounding box annotations in the shapes layer in YOLO format."""

        # Get the current image file name and the corresponding annotation file name
        current_image_file = self.listWidget.currentItem().text()
        annotation_file = os.path.splitext(current_image_file)[0] + ".txt"

        shapes_layer = self.viewer.layers["bbox_layer"]
        print(shapes_layer.feature_defaults)
        print(shapes_layer.features)
        image_layer = self.viewer.layers["image_layer"]
        if image_layer.rgb:
            image_height, image_width, _ = image_layer.data.shape[-3:]
        else:
            image_height, image_width = image_layer.data.shape[-2:]
        shapes_data = shapes_layer.data

        annotations = []

        print(shapes_data)

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
