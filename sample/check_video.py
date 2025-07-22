from napari_simpleannotate._bbox_video_widget import FastVideoReader
import numpy as np
import av


# FastVideoReaderクラスがインポートされていることを前提
# from your_module import FastVideoReader


def test_video_reader(video_path):
    """Test FastVideoReader with specific frames."""
    print(f"Testing video: {video_path}")
    print("-" * 50)

    # Create reader instance
    reader = FastVideoReader(video_path)

    print(f"Video shape: {reader.shape}")
    print(f"Total frames: {reader.total_frames}")
    print(f"FPS: {reader.fps}")
    print("-" * 50)

    # Test specific frames
    test_frames = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]

    for frame_idx in test_frames:
        if frame_idx >= reader.total_frames:
            continue

        print(f"\nTesting frame {frame_idx}:")
        try:
            # Read frame
            frame = reader.read_frame(frame_idx)
            print(f"  Shape: {frame.shape}")
            print(f"  dtype: {frame.dtype}")
            print(f"  Min/Max values: {frame.min()}/{frame.max()}")
            print(f"  Mean value: {frame.mean():.2f}")

            # Check if frame is all black
            if np.all(frame == 0):
                print("  WARNING: Frame is all black!")

            # Test __getitem__ access (what napari uses)
            frame_via_getitem = reader[frame_idx]
            if np.array_equal(frame, frame_via_getitem):
                print("  __getitem__ access: OK")
            else:
                print("  __getitem__ access: MISMATCH!")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Test sequential access
    print("\n" + "-" * 50)
    print("Testing sequential access (frames 0-5):")
    for i in range(6):
        if i >= reader.total_frames:
            break
        frame = reader[i]
        print(f"  Frame {i}: shape={frame.shape}, mean={frame.mean():.2f}")

    # Test slice access
    print("\n" + "-" * 50)
    print("Testing slice access [0:5]:")
    try:
        frames = reader[0:5]
        print(f"  Slice shape: {frames.shape}")
        print(f"  Individual frame means: {[f.mean() for f in frames]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    reader.close()
    print("\n" + "-" * 50)
    print("Test complete")


def compare_with_direct_av(video_path):
    """Compare FastVideoReader output with direct PyAV reading."""
    print(f"\nComparing with direct PyAV reading")
    print("-" * 50)

    # Read with FastVideoReader
    reader = FastVideoReader(video_path)

    # Read with direct PyAV
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Compare first 5 frames
    container.seek(0)
    for i, av_frame in enumerate(container.decode(stream)):
        if i >= 5:
            break

        # Get frame from FastVideoReader
        reader_frame = reader[i]

        # Convert PyAV frame to numpy
        av_array = av_frame.to_ndarray(format="rgb24")

        # Compare
        if np.array_equal(reader_frame, av_array):
            print(f"  Frame {i}: MATCH")
        else:
            print(f"  Frame {i}: MISMATCH")
            print(f"    Reader shape: {reader_frame.shape}, PyAV shape: {av_array.shape}")
            print(f"    Reader mean: {reader_frame.mean():.2f}, PyAV mean: {av_array.mean():.2f}")

    container.close()
    reader.close()


if __name__ == "__main__":
    video_path = "/Users/hiroki/Downloads/no foreign matter/recording_2025-06-10_5A_13_4_non.avi"

    # Run tests
    test_video_reader(video_path)
    compare_with_direct_av(video_path)
