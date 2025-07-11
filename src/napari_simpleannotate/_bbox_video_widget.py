import os
from typing import TYPE_CHECKING

import numpy as np
from skimage import io
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
)
import yaml
from napari_video.napari_video import VideoReaderNP
from napari.utils.notifications import show_warning

from ._utils import find_missing_number, xywh2xyxy

if TYPE_CHECKING:
    import napari


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
            "translation": np.array([0, 0]),
            "size": 10,
            "color": "green",
            "visible": True,
        }
        self.numbers = []
        self.current_class_number = 0
        self.video_path = ""
        self.video_dir = ""
        self.annotation_dir = ""
        self.total_frames = 0
        self.current_frame = 0
        self.video_layer = None
        self.order = 0  # 桁数

    def initLayers(self):
        """Initializes the video and shapes layers in the napari viewer."""
        # Initialize shapes layer with proper features structure
        shapes_layer = self.viewer.add_shapes(
            name="bbox_layer",
            ndim=3,  # 3D for time + 2D coordinates
            text=self.text,
            face_color="transparent",
            edge_color="green",  # デフォルトカラーを設定
            edge_width=2,
        )

        # Initialize empty features
        shapes_layer.features = {"class": [], "frame": []}

        # Connect to frame change event
        self.viewer.dims.events.current_step.connect(self.on_frame_changed)

        # Connect to shape data change events
        shapes_layer.events.data.connect(self.on_shape_added)

    def openVideo(self):
        """Open video file using file dialog."""
        fname = QFileDialog.getOpenFileName(
            self, "Open video file", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )

        if fname[0]:
            self.load_video(fname[0])

    def load_video(self, video_path):
        """Load video file using VideoReaderNP."""
        self.video_path = video_path
        try:
            # VideoReaderNPを使って動画を読み込む
            vr = VideoReaderNP(self.video_path)

            # 既存のビデオレイヤーを削除
            if self.video_layer and self.video_layer in self.viewer.layers:
                self.viewer.layers.remove(self.video_layer)

            self.video_layer = self.viewer.add_image(vr, name="video_layer", rgb=True)
            print(f"Video loaded: {self.video_path}")

            # 動画情報を取得
            self.total_frames = self.video_layer.data.shape[0]
            self.order = len(str(self.total_frames))  # 桁数を取得
            height = self.video_layer.data.shape[1]
            width = self.video_layer.data.shape[2]

            # ディレクトリの設定
            self.video_dir = os.path.dirname(video_path)
            self.annotation_dir = os.path.splitext(video_path)[0]
            if not os.path.exists(self.annotation_dir):
                os.makedirs(self.annotation_dir)
                print(f"Created annotation directory: {self.annotation_dir}")

            # UIの更新
            video_name = os.path.basename(video_path)
            self.video_info_label.setText(f"Video: {video_name} ({self.total_frames} frames, {width}x{height})")
            self.update_frame_info()

            # 既存のデータを読み込む（順序重要：先にクラスを読み込む）
            self.load_classes()

            # shapesレイヤーをリセット
            shapes_layer = self.viewer.layers["bbox_layer"]
            shapes_layer.data = []
            shapes_layer.features = {"class": [], "frame": []}
            shapes_layer.text = self.text

            # アノテーションを読み込む
            self.load_annotations()

            # 最初のフレームに移動
            self.viewer.dims.current_step = (0,) + self.viewer.dims.current_step[1:]

        except Exception as e:
            print(f"Failed to load video: {e}")
            show_warning(f"Failed to load video: {e}")

    def on_frame_changed(self, event):
        """Handle frame change events."""
        if hasattr(event, "value") and len(event.value) > 0:
            new_frame = event.value[0]
            if new_frame != self.current_frame:
                self.current_frame = new_frame
                self.update_frame_info()

    def update_frame_info(self):
        """Update frame information display."""
        self.frame_info_label.setText(f"Frame: {self.current_frame}/{self.total_frames}")

    def on_shape_added(self, event):
        """Handle when a new shape is added."""
        # Check if event has action attribute and if it's relevant
        if hasattr(event, "action"):
            if event.action not in ["adding", "added", "changed"]:
                return

        shapes_layer = self.viewer.layers["bbox_layer"]

        # If no shapes, clear features and return
        if len(shapes_layer.data) == 0:
            shapes_layer.features = {"class": [], "frame": []}
            return

        # Get the currently selected class
        selected_item = self.classlistWidget.selectedItems()
        if selected_item:
            class_text = selected_item[0].text()
        else:
            class_text = "0: Unknown"

        # Get current features as lists
        if hasattr(shapes_layer, "features") and shapes_layer.features is not None:
            if isinstance(shapes_layer.features, dict):
                current_classes = list(shapes_layer.features.get("class", []))
                current_frames = list(shapes_layer.features.get("frame", []))
            else:
                # If features is a DataFrame
                try:
                    current_classes = shapes_layer.features["class"].tolist()
                    current_frames = shapes_layer.features["frame"].tolist()
                except:
                    current_classes = []
                    current_frames = []
        else:
            current_classes = []
            current_frames = []

        # Synchronize features with shapes data
        num_shapes = len(shapes_layer.data)

        # Add features for new shapes
        while len(current_classes) < num_shapes:
            current_classes.append(class_text)
            current_frames.append(self.current_frame)

        # Remove extra features if shapes were deleted
        while len(current_classes) > num_shapes:
            current_classes.pop()
            current_frames.pop()

        # Update features
        shapes_layer.features = {"class": current_classes, "frame": current_frames}

        # Refresh text without triggering another event
        try:
            shapes_layer.refresh_text()
        except:
            pass

    def class_clicked(self):
        """Handle class selection."""
        shapes_layer = self.viewer.layers["bbox_layer"]
        selected_item = self.classlistWidget.selectedItems()
        if not selected_item:
            return

        class_text = selected_item[0].text()

        # Set feature defaults safely
        try:
            if hasattr(shapes_layer, "feature_defaults"):
                shapes_layer.feature_defaults["class"] = class_text
        except:
            pass

        # Update selected shapes with new class
        idxs = list(shapes_layer.selected_data)
        if len(idxs) != 0:
            # Ensure features exist
            if not hasattr(shapes_layer, "features") or shapes_layer.features is None:
                shapes_layer.features = {
                    "class": [class_text] * len(shapes_layer.data),
                    "frame": [self.current_frame] * len(shapes_layer.data),
                }
            else:
                # Get current features and update them
                if isinstance(shapes_layer.features, dict):
                    current_classes = list(shapes_layer.features.get("class", []))
                    current_frames = list(shapes_layer.features.get("frame", []))
                else:
                    try:
                        current_classes = shapes_layer.features["class"].tolist()
                        current_frames = shapes_layer.features["frame"].tolist()
                    except:
                        current_classes = []
                        current_frames = []

                # Ensure lists are long enough
                while len(current_classes) < len(shapes_layer.data):
                    current_classes.append("0: Unknown")
                    current_frames.append(self.current_frame)

                # Update selected indices
                for idx in idxs:
                    if idx < len(current_classes):
                        current_classes[idx] = class_text
                        current_frames[idx] = self.current_frame

                # Update features with new lists
                shapes_layer.features = {"class": current_classes, "frame": current_frames}

            # Refresh text
            try:
                shapes_layer.refresh_text()
            except:
                pass

    def add_class(self):
        """Add new class to the class list."""
        class_name = self.class_textbox.text().strip()
        if not class_name:
            return

        # Check if class already exists
        existing_classes = [self.classlistWidget.item(i).text() for i in range(self.classlistWidget.count())]

        # Check for duplicate class names (ignoring the number prefix)
        for existing in existing_classes:
            if existing.split(": ", 1)[-1] == class_name:
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

        # If this is the first class, select it automatically
        if self.classlistWidget.count() == 1:
            self.classlistWidget.setCurrentRow(0)

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
            QMessageBox.No,
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
        """Load classes from class.yaml file in annotation directory."""
        if not self.annotation_dir:
            return

        class_file_path = os.path.join(self.annotation_dir, "class.yaml")
        try:
            with open(class_file_path, "r") as f:
                class_data = yaml.safe_load(f)

            self.classlistWidget.clear()

            # Handle both old format and new YOLO format
            if isinstance(class_data, dict):
                if "names" in class_data:
                    # New YOLO format
                    names_dict = class_data["names"]
                    for number, class_name in names_dict.items():
                        numbered_class = f"{number}: {class_name}"
                        self.classlistWidget.addItem(numbered_class)
                else:
                    # Old format (direct number: name mapping)
                    for number, class_name in class_data.items():
                        numbered_class = f"{number}: {class_name}"
                        self.classlistWidget.addItem(numbered_class)
                self.sort_classlist()
        except FileNotFoundError:
            pass  # No existing class file
        except Exception as e:
            print(f"Error loading classes: {e}")

    def save_classes(self):
        """Save classes to class.yaml file in annotation directory."""
        if not self.annotation_dir:
            return

        # Create names dictionary for YOLO format
        names = {}
        for i in range(self.classlistWidget.count()):
            item_text = self.classlistWidget.item(i).text()
            number, class_name = item_text.split(":", 1)
            names[int(number.strip())] = class_name.strip()

        # Create YOLO format class data
        class_data = {"names": names}

        class_file_path = os.path.join(self.annotation_dir, "class.yaml")
        try:
            with open(class_file_path, "w") as f:
                yaml.dump(class_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error saving classes: {e}")

    def load_annotations(self):
        """Load existing annotations from individual frame files."""
        if not self.video_path:
            return

        try:
            shapes_layer = self.viewer.layers["bbox_layer"]
            shapes_data = []
            features_data = {"class": [], "frame": []}

            # Get all .txt files in annotation directory
            import glob

            annotation_files = glob.glob(os.path.join(self.annotation_dir, "img*.txt"))

            for annotation_file in annotation_files:
                # Extract frame number from filename
                filename = os.path.basename(annotation_file)
                try:
                    # Remove 'img' prefix and '.txt' suffix to get frame number
                    frame_str = filename[3:-4]  # Skip 'img' and '.txt'
                    frame = int(frame_str)
                except:
                    continue

                with open(annotation_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            # Convert YOLO format to rectangle coordinates
                            # Get video dimensions
                            if self.video_layer:
                                h = self.video_layer.data.shape[1]
                                w = self.video_layer.data.shape[2]
                                bbox_coords = xywh2xyxy([x_center, y_center, width, height], (w, h))

                                # Create rectangle shape [frame, y1, x1, y2, x2]
                                rect = np.array(
                                    [
                                        [frame, bbox_coords[1], bbox_coords[0]],
                                        [frame, bbox_coords[1], bbox_coords[2]],
                                        [frame, bbox_coords[3], bbox_coords[2]],
                                        [frame, bbox_coords[3], bbox_coords[0]],
                                    ]
                                )

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
                # Clear existing shapes
                shapes_layer.data = []
                shapes_layer.features = {"class": [], "frame": []}

                # Add new data
                shapes_layer.data = shapes_data
                shapes_layer.features = features_data

                # Ensure text is properly configured
                shapes_layer.text = self.text

                # Force refresh
                shapes_layer.refresh()
                print(f"Loaded {len(shapes_data)} annotations from {len(annotation_files)} files")

        except Exception as e:
            print(f"Error loading annotations: {e}")
            import traceback

            traceback.print_exc()

    def saveAnnotations(self):
        """Save annotations to individual files for each frame."""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return

        shapes_layer = self.viewer.layers["bbox_layer"]
        if len(shapes_layer.data) == 0:
            QMessageBox.information(self, "Info", "No annotations to save")
            return

        try:
            # Get video dimensions
            if self.video_layer:
                h = self.video_layer.data.shape[1]
                w = self.video_layer.data.shape[2]
            else:
                h, w = 1080, 1920  # Default dimensions if layer not found

            # Get features as lists
            class_list = list(shapes_layer.features.get("class", []))
            frame_list = list(shapes_layer.features.get("frame", []))

            # Group annotations by frame
            frame_annotations = {}

            for i, shape in enumerate(shapes_layer.data):
                if len(shape) == 4:  # Rectangle shape
                    # Extract frame from shape coordinates
                    frame = int(shape[0][0])

                    # Extract bounding box coordinates
                    y_coords = [point[1] for point in shape]
                    x_coords = [point[2] for point in shape]

                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)

                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = abs(x2 - x1) / w
                    height = abs(y2 - y1) / h

                    # Get class information
                    if i < len(class_list):
                        class_text = str(class_list[i])
                        try:
                            class_id = int(class_text.split(":")[0])
                        except:
                            class_id = 0
                    else:
                        class_id = 0

                    # Create annotation line: class_id x_center y_center width height
                    annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                    # Add to frame annotations
                    if frame not in frame_annotations:
                        frame_annotations[frame] = []
                    frame_annotations[frame].append(annotation_line)

            # Save annotations for each frame
            saved_count = 0
            for frame_idx, annotations in frame_annotations.items():
                # Generate annotation filename matching the image filename
                annotation_filename = f"img{str(frame_idx).zfill(self.order)}.txt"
                annotation_path = os.path.join(self.annotation_dir, annotation_filename)

                # Write annotations for this frame
                with open(annotation_path, "w") as f:
                    f.write("\n".join(annotations))
                saved_count += 1
                print(f"Saved annotations for frame {frame_idx} to {annotation_filename}")

            # Save classes in YOLO format
            self.save_classes()

            # Save images for annotated frames
            self.save_annotated_frames()

            QMessageBox.information(
                self, "Success", f"Annotations saved for {saved_count} frames in {self.annotation_dir}"
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save annotations: {str(e)}")

    def save_annotated_frames(self):
        """Save images for frames that have annotations."""
        if not self.video_layer:
            return

        shapes_layer = self.viewer.layers["bbox_layer"]
        if len(shapes_layer.data) == 0:
            return

        # Get unique frame numbers that have annotations
        annotated_frames = set()
        for shape in shapes_layer.data:
            if len(shape) == 4:  # Rectangle shape
                frame = int(shape[0][0])
                annotated_frames.add(frame)

        try:
            for frame_idx in annotated_frames:
                # Generate image filename with zero-padding
                image_filename = f"img{str(frame_idx).zfill(self.order)}.png"
                image_path = os.path.join(self.annotation_dir, image_filename)

                # Skip if image already exists
                if os.path.exists(image_path):
                    continue

                # Read and save frame
                frame_data = self.video_layer.data[int(frame_idx)]
                io.imsave(image_path, frame_data)
                print(f"Saved frame {frame_idx} to {image_filename}")

        except Exception as e:
            print(f"Error saving annotated frames: {e}")
            show_warning(f"Could not save some frame images: {str(e)}")
