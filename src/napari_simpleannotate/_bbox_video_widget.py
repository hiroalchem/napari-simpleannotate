from functools import partial
import os
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
)
import yaml

from ._utils import find_missing_number, xywh2xyxy, save_text

if TYPE_CHECKING:
    import napari

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


class BboxVideoQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.initUI()
        self.initVariables()
        self.initLayers()

    def initUI(self):
        # Create button for opening a video file
        self.open_video_button = QPushButton("Open Video", self)
        self.open_video_button.clicked.connect(self.openVideo)
        
        # Create label to show current video info
        self.video_info_label = QLabel("No video loaded", self)
        
        # Create label to show current frame info
        self.frame_info_label = QLabel("Frame: 0/0", self)

        # Create a list widget for displaying the list of classes
        self.classlistWidget = QListWidget()
        self.classlistWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.classlistWidget.itemClicked.connect(self.class_clicked)

        # Create text box for entering the class names
        self.class_textbox = QLineEdit()
        self.class_textbox.setPlaceholderText("Enter class name")

        # Create button for adding class to classlist
        self.add_class_button = QPushButton("Add class", self)
        self.add_class_button.clicked.connect(self.add_class)

        # Create button for deleting class from classlist
        self.del_class_button = QPushButton("Delete selected class", self)
        self.del_class_button.clicked.connect(self.del_class)

        # Create button for saving the bounding box annotations
        self.save_button = QPushButton("Save Annotations", self)
        self.save_button.clicked.connect(self.saveAnnotations)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.open_video_button)
        layout.addWidget(self.video_info_label)
        layout.addWidget(self.frame_info_label)
        layout.addWidget(QLabel("Classes:"))
        layout.addWidget(self.classlistWidget)
        layout.addWidget(self.class_textbox)
        layout.addWidget(self.add_class_button)
        layout.addWidget(self.del_class_button)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def initVariables(self):
        """Initializes the variables."""
        self.features = {"class": [], "frame": []}
        self.text = {
            "string": "{class}",
            "anchor": "upper_left",
            "translation": [0, 0],
            "size": 10,
            "color": "green",
        }
        self.numbers = []
        self.current_class_number = 0
        self.video_path = ""
        self.video_dir = ""
        self.total_frames = 0
        self.current_frame = 0

    def initLayers(self):
        """Initializes the video and shapes layers in the napari viewer."""
        # Video layer will be added when video is loaded
        self.viewer.add_shapes(name="bbox_layer", features=self.features, text=self.text)
        
        # Connect to frame change event
        self.viewer.dims.events.current_step.connect(self.on_frame_changed)

    def openVideo(self):
        """Open video file using file dialog."""
        if not HAS_PYAV:
            QMessageBox.warning(self, "Error", "PyAV is required to load videos.\nPlease install it with: pip install av")
            return
            
        fname = QFileDialog.getOpenFileName(
            self, 
            "Open video file", 
            "/",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        
        if fname[0]:
            self.load_video(fname[0])

    def load_video(self, video_path):
        """Load video file and initialize annotations."""
        try:
            # Load video using PyAV
            video_frames = self.load_video_with_pyav(video_path)
            
            # Remove existing video layers
            layers_to_remove = []
            for layer in self.viewer.layers:
                if hasattr(layer, 'data') and hasattr(layer.data, 'shape') and len(layer.data.shape) > 3:
                    layers_to_remove.append(layer.name)
            
            for layer_name in layers_to_remove:
                if layer_name in self.viewer.layers:
                    del self.viewer.layers[layer_name]
            
            # Add video layer (as time series)
            self.viewer.add_image(video_frames, name="video_layer")
            
            # Store video information
            self.video_path = video_path
            self.video_dir = os.path.dirname(video_path)
            self.total_frames = len(video_frames)
            self.current_frame = 0
            
            # Update UI
            video_name = os.path.basename(video_path)
            self.video_info_label.setText(f"Video: {video_name}")
            self.update_frame_info()
            
            # Load existing classes if available
            self.load_classes()
            
            # Load existing annotations if available
            self.load_annotations()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")

    def load_video_with_pyav(self, video_path):
        """Load video frames using PyAV."""
        container = av.open(video_path)
        frames = []
        
        for frame in container.decode(video=0):
            # Convert frame to numpy array
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        
        container.close()
        
        # Convert to numpy array with shape (T, H, W, C)
        return np.array(frames)

    def on_frame_changed(self, event):
        """Handle frame change events."""
        if hasattr(event, 'value') and len(event.value) > 0:
            self.current_frame = event.value[0]
            self.update_frame_info()

    def update_frame_info(self):
        """Update frame information display."""
        self.frame_info_label.setText(f"Frame: {self.current_frame}/{self.total_frames}")

    def class_clicked(self):
        """Handle class selection."""
        shapes_layer = self.viewer.layers["bbox_layer"]
        selected_item = self.classlistWidget.selectedItems()
        if not selected_item:
            return
        
        class_text = selected_item[0].text()
        shapes_layer.feature_defaults["class"] = class_text
        
        # Update selected shapes with new class
        idxs = list(shapes_layer.selected_data)
        if len(idxs) != 0:
            shapes_layer.features.loc[idxs, "class"] = class_text
            # Update frame information for selected shapes
            shapes_layer.features.loc[idxs, "frame"] = self.current_frame
            shapes_layer.refresh_text()

    def add_class(self):
        """Add new class to the class list."""
        class_name = self.class_textbox.text().strip()
        if not class_name:
            return
        
        # Check if class already exists
        existing_classes = [self.classlistWidget.item(i).text() for i in range(self.classlistWidget.count())]
        if class_name in existing_classes:
            QMessageBox.information(self, "Info", "Class already exists")
            return
        
        # Find next available number
        if len(existing_classes) == 0:
            self.current_class_number = 0
        else:
            self.numbers = [int(name.split(":")[0]) for name in existing_classes]
            self.current_class_number = find_missing_number(self.numbers)
        
        # Add numbered class
        numbered_class = f"{self.current_class_number}: {class_name}"
        self.classlistWidget.addItem(numbered_class)
        self.sort_classlist()
        self.class_textbox.clear()
        
        # Save classes to file
        self.save_classes()

    def del_class(self):
        """Delete selected class from the class list."""
        selected_items = self.classlistWidget.selectedItems()
        if not selected_items:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion", 
            "Do you want to delete the selected class?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            selected_item = selected_items[0]
            self.classlistWidget.takeItem(self.classlistWidget.row(selected_item))
            self.save_classes()

    def sort_classlist(self):
        """Sort class list by number."""
        items_text = [self.classlistWidget.item(i).text() for i in range(self.classlistWidget.count())]
        
        def extract_number(item_text):
            return int(item_text.split(":")[0].strip())
        
        sorted_items_text = sorted(items_text, key=extract_number)
        
        self.classlistWidget.clear()
        for item_text in sorted_items_text:
            self.classlistWidget.addItem(item_text)

    def load_classes(self):
        """Load classes from class.yaml file in video directory."""
        if not self.video_dir:
            return
        
        class_file_path = os.path.join(self.video_dir, "class.yaml")
        try:
            with open(class_file_path, 'r') as f:
                class_data = yaml.safe_load(f)
            
            self.classlistWidget.clear()
            if isinstance(class_data, dict):
                for number, class_name in class_data.items():
                    numbered_class = f"{number}: {class_name}"
                    self.classlistWidget.addItem(numbered_class)
                self.sort_classlist()
        except FileNotFoundError:
            pass  # No existing class file
        except Exception as e:
            print(f"Error loading classes: {e}")

    def save_classes(self):
        """Save classes to class.yaml file in video directory."""
        if not self.video_dir:
            return
        
        class_data = {}
        for i in range(self.classlistWidget.count()):
            item_text = self.classlistWidget.item(i).text()
            number, class_name = item_text.split(":", 1)
            class_data[int(number.strip())] = class_name.strip()
        
        class_file_path = os.path.join(self.video_dir, "class.yaml")
        try:
            with open(class_file_path, 'w') as f:
                yaml.dump(class_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving classes: {e}")

    def load_annotations(self):
        """Load existing annotations for the video."""
        if not self.video_path:
            return
        
        # Generate annotation file path (video_name.txt)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        annotation_file = os.path.join(self.video_dir, f"{video_name}.txt")
        
        try:
            shapes_layer = self.viewer.layers["bbox_layer"]
            shapes_data = []
            features_data = {"class": [], "frame": []}
            
            with open(annotation_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 6:  # class_id, frame, x_center, y_center, width, height
                        class_id = int(parts[0])
                        frame = int(parts[1])
                        x_center, y_center, width, height = map(float, parts[2:6])
                        
                        # Convert YOLO format to rectangle coordinates
                        # Assuming video dimensions (you might need to get actual dimensions)
                        if hasattr(self.viewer.layers["video_layer"], 'data'):
                            h, w = self.viewer.layers["video_layer"].data.shape[-2:]
                            bbox_coords = xywh2xyxy([x_center, y_center, width, height], (w, h))
                            
                            # Create rectangle shape [frame, y1, x1, y2, x2]
                            rect = np.array([
                                [frame, bbox_coords[1], bbox_coords[0]],
                                [frame, bbox_coords[1], bbox_coords[2]],
                                [frame, bbox_coords[3], bbox_coords[2]],
                                [frame, bbox_coords[3], bbox_coords[0]]
                            ])
                            
                            shapes_data.append(rect)
                            
                            # Find class name from class list
                            class_name = f"{class_id}: Unknown"
                            for i in range(self.classlistWidget.count()):
                                item_text = self.classlistWidget.item(i).text()
                                if item_text.startswith(f"{class_id}:"):
                                    class_name = item_text
                                    break
                            
                            features_data["class"].append(class_name)
                            features_data["frame"].append(frame)
            
            if shapes_data:
                shapes_layer.data = shapes_data
                shapes_layer.features = features_data
                shapes_layer.refresh()
                
        except FileNotFoundError:
            pass  # No existing annotation file
        except Exception as e:
            print(f"Error loading annotations: {e}")

    def saveAnnotations(self):
        """Save annotations to file."""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        
        shapes_layer = self.viewer.layers["bbox_layer"]
        if len(shapes_layer.data) == 0:
            QMessageBox.information(self, "Info", "No annotations to save")
            return
        
        # Generate annotation file path
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        annotation_file = os.path.join(self.video_dir, f"{video_name}.txt")
        
        try:
            annotations = []
            video_layer = self.viewer.layers["video_layer"]
            h, w = video_layer.data.shape[-2:]
            
            for i, shape in enumerate(shapes_layer.data):
                if len(shape) == 4:  # Rectangle shape
                    # Extract frame from shape coordinates
                    frame = int(shape[0][0])
                    
                    # Extract bounding box coordinates
                    y_coords = [point[1] for point in shape]
                    x_coords = [point[2] for point in shape]
                    
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    # Convert to YOLO format
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = abs(x2 - x1) / w
                    height = abs(y2 - y1) / h
                    
                    # Get class information
                    class_text = shapes_layer.features["class"][i]
                    class_id = int(class_text.split(":")[0])
                    
                    # Create annotation line: class_id frame x_center y_center width height
                    annotation_line = f"{class_id} {frame} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    annotations.append(annotation_line)
            
            # Save annotations
            with open(annotation_file, 'w') as f:
                f.write('\n'.join(annotations))
            
            # Save classes
            self.save_classes()
            
            QMessageBox.information(self, "Success", f"Annotations saved to {annotation_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotations: {str(e)}")