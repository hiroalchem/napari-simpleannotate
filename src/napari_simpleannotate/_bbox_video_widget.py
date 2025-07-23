"""
BboxVideoQWidget - Video Annotation Widget for napari-simpleannotate

PERFORMANCE OPTIMIZATION STATUS:
- ✅ LRU Frame Cache: Implemented with configurable size (default: 50 frames)
- ✅ Parallel Prefetching: Multi-threaded frame preloading with separate VideoReader instances
- ❌ ZARR Support: DISABLED - Available for future improvement

ZARR FUNCTIONALITY (Currently Disabled):
The zarr-based video loading functionality has been temporarily disabled due to:
1. Slow conversion speeds for large videos (4K+)
2. High memory usage during conversion process
3. Blosc compressor compatibility issues with different zarr format versions
4. Need for better chunk size calculation algorithms

Current solution uses OpenCV-based VideoReaderNP with:
- LRU cache for recently accessed frames
- Parallel prefetching using separate VideoReader instances per thread
- Thread-safe frame loading with proper synchronization

TODO for future zarr re-enablement:
- Optimize conversion speed using better algorithms
- Implement streaming conversion to reduce memory usage
- Resolve blosc/zarr format compatibility
- Add better progress reporting and cancellation support
"""

import multiprocessing as mp
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import yaml
import zarr
from napari.utils.notifications import show_warning
from napari_video.napari_video import VideoReaderNP
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage import io

from ._utils import find_missing_number, xywh2xyxy

if TYPE_CHECKING:
    pass


class CachedVideoReader:
    """Video reader wrapper with LRU cache and parallel prefetching."""

    def __init__(self, video_reader, max_cache_size=100, prefetch_size=10, video_path=None):
        self.video_reader = video_reader
        self.video_path = video_path  # Store path for creating new VideoReader instances
        self.max_cache_size = max_cache_size
        self.prefetch_size = prefetch_size
        self.cache = OrderedDict()
        self.cache_lock = threading.RLock()
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_futures = {}
        self.total_frames = len(video_reader) if hasattr(video_reader, "__len__") else video_reader.shape[0]

        # Delegate attributes to original video reader
        for attr in ["shape", "dtype", "ndim", "size", "nbytes"]:
            if hasattr(video_reader, attr):
                setattr(self, attr, getattr(video_reader, attr))

        # Make the object hashable
        self._hash = id(self)

        # Add array interface for better numpy/napari compatibility
        if hasattr(video_reader, "__array_interface__"):
            self.__array_interface__ = video_reader.__array_interface__

    def __len__(self):
        return self.total_frames

    def __array__(self):
        """Convert to numpy array when needed."""
        # Return all frames as numpy array
        frames = [self.get_frame(i) for i in range(len(self))]
        return np.stack(frames)

    def __hash__(self):
        """Make the object hashable."""
        return self._hash

    def __eq__(self, other):
        """Equality comparison."""
        # Handle comparison with unhashable types like slices
        try:
            if self is other:
                return True
            # Don't compare with slices or other non-CachedVideoReader objects
            if not isinstance(other, CachedVideoReader):
                return False
            return False
        except TypeError:
            # If comparison fails (e.g., with unhashable types), return False
            return False

    def __getitem__(self, index):
        """Get frame(s) with caching."""
        # Handle tuple of slices (from napari)
        if isinstance(index, tuple):
            # Extract frame index from first element
            frame_slice = index[0]

            if isinstance(frame_slice, slice):
                # Handle slice access for frames
                indices = range(*frame_slice.indices(len(self)))
                frames = []
                for i in indices:
                    frame = self.get_frame(i)
                    # Apply remaining slices to the frame
                    if len(index) > 1:
                        frame = frame[index[1:]]
                    frames.append(frame)

                # Return as numpy array
                if frames:
                    return np.stack(frames)
                else:
                    # Return empty array with proper shape
                    shape = list(self.shape)
                    shape[0] = 0
                    if len(index) > 1:
                        # Adjust shape based on slicing
                        for i, s in enumerate(index[1:], 1):
                            if isinstance(s, slice):
                                shape[i] = len(range(*s.indices(shape[i])))
                            elif isinstance(s, int):
                                shape.pop(i)
                    return np.zeros(shape, dtype=self.dtype)
            elif isinstance(frame_slice, (int, np.integer)):
                # Single frame access
                frame = self.get_frame(int(frame_slice))
                # Apply remaining slices
                if len(index) > 1:
                    frame = frame[index[1:]]
                return frame
            else:
                raise TypeError(f"Invalid index type: {type(frame_slice)}")

        elif isinstance(index, slice):
            # Handle simple slice access
            indices = range(*index.indices(len(self)))
            frames = [self.get_frame(i) for i in indices]
            # Return as numpy array for napari compatibility
            if frames:
                return np.stack(frames)
            else:
                # Return empty array with proper shape
                shape = list(self.shape)
                shape[0] = 0
                return np.zeros(shape, dtype=self.dtype)
        elif isinstance(index, (int, np.integer)):
            # Handle single frame access
            return self.get_frame(int(index))
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def get_frame(self, frame_idx):
        """Get frame with caching and prefetching."""
        with self.cache_lock:
            # Check if frame is already cached
            if frame_idx in self.cache:
                # Move to end (most recently used)
                frame = self.cache.pop(frame_idx)
                self.cache[frame_idx] = frame

                # Trigger prefetch for nearby frames
                self._prefetch_nearby_frames(frame_idx)

                return frame

        # Frame not in cache, load it
        frame = self._load_frame(frame_idx)

        with self.cache_lock:
            # Add to cache
            self.cache[frame_idx] = frame

            # Remove oldest frames if cache is full
            while len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)

        # Trigger prefetch for nearby frames
        self._prefetch_nearby_frames(frame_idx)

        return frame

    def _load_frame(self, frame_idx):
        """Load a single frame from video reader."""
        try:
            return self.video_reader[frame_idx].copy()
        except (IndexError, ValueError, OSError) as e:
            print(f"Error loading frame {frame_idx}: {e}")
            # Return black frame as fallback
            shape = getattr(self.video_reader, "shape", (1, 480, 640, 3))
            return np.zeros(shape[1:], dtype="uint8")

    def _prefetch_nearby_frames(self, center_frame):
        """Prefetch frames around the current frame."""
        # Calculate prefetch range
        start_frame = max(0, center_frame - self.prefetch_size // 2)
        end_frame = min(self.total_frames, center_frame + self.prefetch_size // 2 + 1)

        for frame_idx in range(start_frame, end_frame):
            if frame_idx != center_frame and frame_idx not in self.cache:
                # Cancel any existing prefetch for this frame
                if frame_idx in self.prefetch_futures:
                    self.prefetch_futures[frame_idx].cancel()

                # Start new prefetch with separate VideoReader instance
                if self.prefetch_executor:
                    future = self.prefetch_executor.submit(self._prefetch_frame_safe, frame_idx)
                    self.prefetch_futures[frame_idx] = future

    def _prefetch_frame_safe(self, frame_idx):
        """Prefetch a frame using a separate VideoReader instance."""
        try:
            # Create a separate VideoReader for this thread to avoid conflicts
            if self.video_path:
                thread_reader = VideoReaderNP(self.video_path)
                frame = thread_reader[frame_idx].copy()
            else:
                # Fallback to main reader (might cause threading issues)
                frame = self._load_frame(frame_idx)

            with self.cache_lock:
                # Only add if not already in cache and cache isn't full
                if frame_idx not in self.cache and len(self.cache) < self.max_cache_size:
                    self.cache[frame_idx] = frame

                    # Remove oldest frames if cache is full
                    while len(self.cache) > self.max_cache_size:
                        self.cache.popitem(last=False)

            # Clean up future reference
            if frame_idx in self.prefetch_futures:
                del self.prefetch_futures[frame_idx]

        except Exception as e:
            print(f"Safe prefetch error for frame {frame_idx}: {e}")

    def _prefetch_frame(self, frame_idx):
        """Prefetch a frame in background thread."""
        try:
            frame = self._load_frame(frame_idx)

            with self.cache_lock:
                # Only add if not already in cache and cache isn't full
                if frame_idx not in self.cache and len(self.cache) < self.max_cache_size:
                    self.cache[frame_idx] = frame

                    # Remove oldest frames if cache is full
                    while len(self.cache) > self.max_cache_size:
                        self.cache.popitem(last=False)

            # Clean up future reference
            if frame_idx in self.prefetch_futures:
                del self.prefetch_futures[frame_idx]

        except Exception as e:
            print(f"Prefetch error for frame {frame_idx}: {e}")

    def clear_cache(self):
        """Clear all cached frames."""
        with self.cache_lock:
            self.cache.clear()

        # Cancel all prefetch operations
        for future in self.prefetch_futures.values():
            future.cancel()
        self.prefetch_futures.clear()

    def get_cache_info(self):
        """Get cache statistics."""
        with self.cache_lock:
            return {
                "cache_size": len(self.cache),
                "max_cache_size": self.max_cache_size,
                "active_prefetches": len(self.prefetch_futures),
            }


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

        # Create checkbox for zarr conversion (DISABLED for now - future improvement)
        # TODO: Re-enable zarr functionality after resolving conversion speed and compatibility issues
        self.use_zarr_checkbox = QCheckBox("Use Zarr for faster loading (Experimental - Disabled)", self)
        self.use_zarr_checkbox.setChecked(False)  # Always disabled for now
        self.use_zarr_checkbox.setEnabled(False)  # Disable the checkbox
        self.use_zarr_checkbox.setToolTip(
            "Convert video to Zarr format for memory-efficient fast loading\n(Currently disabled - using frame cache instead)"
        )

        # Create progress bar for zarr conversion (hidden when zarr is disabled)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)  # Keep hidden since zarr is disabled

        # Create label to show current video info
        self.video_info_label = QLabel("No video loaded", self)

        # Create label to show current frame info
        self.frame_info_label = QLabel("Frame: 0/0", self)

        # Create label to show cache info
        self.cache_info_label = QLabel("Cache: disabled", self)

        # Create navigation buttons for jumping to annotations
        self.jump_prev_button = QPushButton("← Previous Annotation (Q)", self)
        self.jump_prev_button.clicked.connect(self.jump_to_previous_annotation)
        self.jump_prev_button.setEnabled(False)

        self.jump_next_button = QPushButton("Next Annotation (W) →", self)
        self.jump_next_button.clicked.connect(self.jump_to_next_annotation)
        self.jump_next_button.setEnabled(False)

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
        layout.addWidget(self.use_zarr_checkbox)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.video_info_label)
        layout.addWidget(self.frame_info_label)
        layout.addWidget(self.cache_info_label)
        layout.addWidget(self.jump_prev_button)
        layout.addWidget(self.jump_next_button)
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
            "translation": np.array([0, 0, 0]),
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
        self.frame_cache = None  # Video frame cache

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

        # Bind keyboard shortcuts for navigation
        self.viewer.bind_key("q", self.jump_to_previous_annotation)
        self.viewer.bind_key("w", self.jump_to_next_annotation)

    def openVideo(self):
        """Open video file using file dialog."""
        fname = QFileDialog.getOpenFileName(
            self, "Open video file", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )

        if fname[0]:
            self.load_video(fname[0])

    def convert_video_to_zarr(self, video_path, zarr_path):
        """Fast video to zarr conversion with multiprocessing.

        NOTE: This method is currently DISABLED - zarr functionality is commented out
        for future improvement. Issues to resolve before re-enabling:
        1. Slow conversion speed for large videos
        2. High memory usage during conversion
        3. Blosc compressor compatibility issues
        4. Better chunk size calculation algorithms

        Currently using LRU cache with parallel prefetching instead.
        """
        import threading
        from queue import Queue

        import cv2

        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Open video to get properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"Video info: {total_frames} frames, {width}x{height}, {fps} fps")

            # Read first frame to determine channels
            ret, frame = cap.read()
            if not ret:
                raise Exception("Could not read first frame")
            channels = frame.shape[2] if len(frame.shape) > 2 else 1
            cap.release()

            # Calculate safe chunk size
            bytes_per_frame = height * width * channels
            max_bytes = 1024 * 1024 * 1024  # 1GB limit
            max_frames_per_chunk = max(1, max_bytes // bytes_per_frame)
            chunk_frames = min(10, max_frames_per_chunk, total_frames)

            print(f"Using chunk size of {chunk_frames} frames, processing with {mp.cpu_count()} workers")

            # Create zarr array
            from numcodecs import Blosc

            compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)  # Faster compression

            z = zarr.open_array(
                zarr_path,
                mode="w",
                shape=(total_frames, height, width, channels),
                chunks=(chunk_frames, height, width, channels),
                dtype="uint8",
                compressor=compressor,
                zarr_format=2,
            )

            # Producer-consumer pattern with threading
            frame_queue = Queue(maxsize=100)  # Buffer for frames
            frames_processed = 0

            def video_reader():
                """Read frames in a separate thread."""
                cap = cv2.VideoCapture(video_path)
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_queue.put((i, frame_rgb))
                    else:
                        frame_queue.put((i, None))
                frame_queue.put(None)  # Signal end
                cap.release()

            # Start video reader thread
            reader_thread = threading.Thread(target=video_reader)
            reader_thread.start()

            # Process frames with thread pool for parallel color conversion
            batch_frames = []
            batch_indices = []

            with ThreadPoolExecutor(max_workers=4) as executor:
                while True:
                    item = frame_queue.get()
                    if item is None:  # End signal
                        break

                    idx, frame = item
                    if frame is not None:
                        batch_frames.append(frame)
                        batch_indices.append(idx)

                    # Write batch when full or at end
                    if len(batch_frames) >= chunk_frames or idx == total_frames - 1:
                        if batch_frames:
                            # Write to zarr
                            batch_array = np.stack(batch_frames)
                            start_idx = batch_indices[0]
                            z[start_idx : start_idx + len(batch_frames)] = batch_array

                            frames_processed += len(batch_frames)
                            batch_frames = []
                            batch_indices = []

                            # Update progress
                            progress = int(frames_processed / total_frames * 100)
                            self.progress_bar.setValue(progress)

                            from qtpy.QtWidgets import QApplication

                            QApplication.processEvents()

            reader_thread.join()

            # Hide progress bar
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)

            print(f"Successfully converted {frames_processed} frames to zarr: {zarr_path}")
            return True

        except Exception as e:
            self.progress_bar.setVisible(False)
            print(f"Error in zarr conversion: {e}")
            import traceback

            traceback.print_exc()

            # Fallback to simple method
            print("Trying fallback method...")
            return self.convert_video_to_zarr_simple(video_path, zarr_path)

    def convert_video_to_zarr_simple(self, video_path, zarr_path):
        """Simple conversion without multiprocessing."""
        import cv2

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            # Get properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False
            channels = frame.shape[2] if len(frame.shape) > 2 else 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Create zarr with minimal compression for speed
            z = zarr.open_array(
                zarr_path,
                mode="w",
                shape=(total_frames, height, width, channels),
                chunks=(1, height, width, channels),
                dtype="uint8",
                compressor=None,  # No compression for maximum speed
                zarr_format=2,
            )

            # Simple sequential processing
            for i in range(total_frames):
                ret, frame = cap.read()
                if ret:
                    z[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if i % 50 == 0:
                    self.progress_bar.setValue(int((i + 1) / total_frames * 100))
                    from qtpy.QtWidgets import QApplication

                    QApplication.processEvents()

            cap.release()
            self.progress_bar.setVisible(False)

            print(f"Converted {total_frames} frames (uncompressed)")
            return True

        except Exception as e:
            self.progress_bar.setVisible(False)
            print(f"Simple conversion failed: {e}")
            return self.convert_video_to_zarr_fallback(video_path, zarr_path)

    def convert_video_to_zarr_fallback(self, video_path, zarr_path):
        """Fallback conversion method using VideoReaderNP."""
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Read video with VideoReaderNP
            vr = VideoReaderNP(video_path)
            total_frames = vr.shape[0]
            height = vr.shape[1]
            width = vr.shape[2]
            channels = vr.shape[3] if len(vr.shape) > 3 else 3

            # Create zarr array
            from numcodecs import Blosc

            compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

            z = zarr.open_array(
                zarr_path,
                mode="w",
                shape=(total_frames, height, width, channels),
                chunks=(1, height, width, channels),
                dtype="uint8",
                compressor=compressor,
                zarr_format=2,
            )

            # Process frames
            for i in range(total_frames):
                try:
                    z[i] = vr[i]
                except Exception as e:
                    print(f"Error reading frame {i}: {e}")
                    z[i] = np.zeros((height, width, channels), dtype="uint8")

                if i % 10 == 0 or i == total_frames - 1:
                    progress = int((i + 1) / total_frames * 100)
                    self.progress_bar.setValue(progress)

                    from qtpy.QtWidgets import QApplication

                    QApplication.processEvents()

            self.progress_bar.setVisible(False)
            print(f"Successfully converted video to zarr format: {zarr_path}")
            return True

        except Exception as e:
            self.progress_bar.setVisible(False)
            print(f"Error in fallback conversion: {e}")
            return False

    def load_video(self, video_path):
        """Load video file using VideoReaderNP or zarr if available, with caching."""
        self.video_path = video_path
        try:
            # Clear existing cache
            if self.frame_cache:
                self.frame_cache.clear_cache()

            # ZARR FUNCTIONALITY DISABLED FOR NOW - FUTURE IMPROVEMENT
            # TODO: Re-enable after resolving performance and compatibility issues
            # Issues to resolve:
            # 1. Slow conversion speed for large videos
            # 2. Memory usage during conversion
            # 3. Blosc compressor compatibility with zarr format versions
            # 4. Better chunk size calculation for different video resolutions

            # Keep zarr_path definition for future use
            zarr_path = os.path.splitext(video_path)[0] + ".zarr"

            # Force use of cached VideoReaderNP (zarr checkbox is disabled)
            print("Using cached VideoReaderNP with LRU cache and parallel prefetching")
            base_reader = VideoReaderNP(self.video_path)
            video_data = CachedVideoReader(base_reader, max_cache_size=50, prefetch_size=20, video_path=self.video_path)
            self.frame_cache = video_data

            # COMMENTED OUT ZARR CODE - KEEP FOR FUTURE REFERENCE
            # if self.use_zarr_checkbox.isChecked():
            #     # Check if zarr file exists
            #     if os.path.exists(zarr_path):
            #         print(f"Loading existing zarr file: {zarr_path}")
            #         video_data = zarr.open_array(zarr_path, mode='r')
            #     else:
            #         # Convert video to zarr
            #         print(f"Converting video to zarr format...")
            #         success = self.convert_video_to_zarr(video_path, zarr_path)
            #         if success:
            #             video_data = zarr.open_array(zarr_path, mode='r')
            #         else:
            #             # Fallback to VideoReaderNP with cache
            #             print("Zarr conversion failed, using cached VideoReaderNP")
            #             base_reader = VideoReaderNP(self.video_path)
            #             video_data = CachedVideoReader(base_reader, max_cache_size=50, prefetch_size=20, video_path=self.video_path)
            #             self.frame_cache = video_data
            # else:
            #     # Use cached VideoReaderNP
            #     print("Using cached VideoReaderNP for faster seeking")
            #     base_reader = VideoReaderNP(self.video_path)
            #     video_data = CachedVideoReader(base_reader, max_cache_size=50, prefetch_size=20, video_path=self.video_path)
            #     self.frame_cache = video_data

            # 既存のビデオレイヤーを削除
            if self.video_layer:
                try:
                    if self.video_layer in self.viewer.layers:
                        self.viewer.layers.remove(self.video_layer)
                except:
                    # If comparison fails, remove by name
                    for layer in list(self.viewer.layers):
                        if layer.name == "video_layer":
                            self.viewer.layers.remove(layer)
                            break

            # Debug info
            print(f"Video data type: {type(video_data)}")
            print(f"Video data shape: {getattr(video_data, 'shape', 'No shape attr')}")

            try:
                self.video_layer = self.viewer.add_image(video_data, name="video_layer", rgb=True)
                print(f"Video loaded: {self.video_path}")
            except Exception as e:
                import traceback

                print(f"Error adding image layer: {e}")
                print("Full traceback:")
                traceback.print_exc()
                raise

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

            # Enable navigation buttons
            self.jump_prev_button.setEnabled(True)
            self.jump_next_button.setEnabled(True)

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

        # Update cache info if cache is active
        if self.frame_cache:
            cache_info = self.frame_cache.get_cache_info()
            self.cache_info_label.setText(
                f"Cache: {cache_info['cache_size']}/{cache_info['max_cache_size']} "
                f"(prefetch: {cache_info['active_prefetches']})"
            )
        else:
            self.cache_info_label.setText("Cache: disabled")

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
            with open(class_file_path) as f:
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

                with open(annotation_file) as f:
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

    def jump_to_previous_annotation(self, viewer=None):
        """Jump to the nearest annotation before the current frame."""
        if not self.video_layer:
            return

        shapes_layer = self.viewer.layers["bbox_layer"]
        if len(shapes_layer.data) == 0:
            return

        # Get current frame
        current_frame = self.current_frame

        # Get all annotated frames
        annotated_frames = set()
        for shape in shapes_layer.data:
            if len(shape) == 4:  # Rectangle shape
                frame = int(shape[0][0])
                annotated_frames.add(frame)

        # Find previous annotated frames
        previous_frames = [f for f in annotated_frames if f < current_frame]

        if previous_frames:
            # Jump to the nearest previous frame
            target_frame = max(previous_frames)
            self.viewer.dims.current_step = (target_frame,) + self.viewer.dims.current_step[1:]
            print(f"Jumped to previous annotation at frame {target_frame}")
        else:
            print("No previous annotations found")

    def jump_to_next_annotation(self, viewer=None):
        """Jump to the nearest annotation after the current frame."""
        if not self.video_layer:
            return

        shapes_layer = self.viewer.layers["bbox_layer"]
        if len(shapes_layer.data) == 0:
            return

        # Get current frame
        current_frame = self.current_frame

        # Get all annotated frames
        annotated_frames = set()
        for shape in shapes_layer.data:
            if len(shape) == 4:  # Rectangle shape
                frame = int(shape[0][0])
                annotated_frames.add(frame)

        # Find next annotated frames
        next_frames = [f for f in annotated_frames if f > current_frame]

        if next_frames:
            # Jump to the nearest next frame
            target_frame = min(next_frames)
            self.viewer.dims.current_step = (target_frame,) + self.viewer.dims.current_step[1:]
            print(f"Jumped to next annotation at frame {target_frame}")
        else:
            print("No next annotations found")
