import os
import tempfile

import pytest
import yaml

from napari_simpleannotate._utils import (
    find_missing_number,
    save_text,
    xywh2xyxy,
)


def test_find_missing_number_empty_list():
    """Test find_missing_number with empty list."""
    result = find_missing_number([])
    assert result == 0


def test_find_missing_number_sequential():
    """Test find_missing_number with sequential numbers."""
    result = find_missing_number([0, 1, 2, 3])
    assert result == 4


def test_find_missing_number_missing_start():
    """Test find_missing_number when 0 is missing."""
    result = find_missing_number([1, 2, 3])
    assert result == 0


def test_find_missing_number_missing_middle():
    """Test find_missing_number with gap in middle."""
    result = find_missing_number([0, 1, 3, 4])
    assert result == 2


def test_find_missing_number_unsorted():
    """Test find_missing_number with unsorted input."""
    result = find_missing_number([3, 0, 1, 4])
    assert result == 2


def test_find_missing_number_single_element():
    """Test find_missing_number with single element."""
    result = find_missing_number([0])
    assert result == 1

    result = find_missing_number([5])
    assert result == 0


def test_xywh2xyxy_basic():
    """Test basic xywh2xyxy conversion."""
    # Center at (0.5, 0.5), width=0.2, height=0.3, scale=(100, 100)
    xywh = [0.5, 0.5, 0.2, 0.3]
    scale = (100, 100)

    result = xywh2xyxy(xywh, scale)

    # Expected: x1=40, y1=35, x2=60, y2=65
    expected = [40.0, 35.0, 60.0, 65.0]
    # Use pytest.approx for floating point comparison
    import pytest

    assert result == pytest.approx(expected)


def test_xywh2xyxy_different_scales():
    """Test xywh2xyxy with different x and y scales."""
    xywh = [0.5, 0.5, 0.4, 0.2]
    scale = (200, 100)  # width=200, height=100

    result = xywh2xyxy(xywh, scale)

    # Expected: x1=60, y1=40, x2=140, y2=60
    expected = [60.0, 40.0, 140.0, 60.0]
    import pytest

    assert result == pytest.approx(expected)


def test_xywh2xyxy_edge_cases():
    """Test xywh2xyxy with edge cases."""
    # Zero width and height
    result = xywh2xyxy([0.5, 0.5, 0, 0], (100, 100))
    expected = [50.0, 50.0, 50.0, 50.0]
    assert result == expected

    # Full width and height
    result = xywh2xyxy([0.5, 0.5, 1.0, 1.0], (100, 100))
    expected = [0.0, 0.0, 100.0, 100.0]
    assert result == expected


def test_save_text_annotations():
    """Test saving text as annotations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test.txt")
        text_content = "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2"

        save_text(filepath, text_content, "annotations")

        # Check file was created and content is correct
        assert os.path.exists(filepath)
        with open(filepath) as f:
            saved_content = f.read()
        assert saved_content == text_content


def test_save_text_classlist():
    """Test saving text as classlist (YAML)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "class.yaml")
        class_data = {0: "person", 1: "car", 2: "bike"}

        save_text(filepath, class_data, "classlist")

        # Check file was created and content is correct
        assert os.path.exists(filepath)
        with open(filepath) as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == class_data


def test_save_text_invalid_type():
    """Test save_text with invalid file type."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test.txt")

        with pytest.raises(ValueError, match="Invalid file_type"):
            save_text(filepath, "content", "invalid_type")


def test_save_text_classlist_complex():
    """Test saving complex classlist data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "complex_class.yaml")
        complex_data = {0: "person", 1: "vehicle", 5: "animal", 10: "object"}  # Non-sequential

        save_text(filepath, complex_data, "classlist")

        # Verify the saved content
        with open(filepath) as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == complex_data


def test_save_text_annotations_multiline():
    """Test saving multi-line annotation text."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "annotations.txt")
        annotations = ["0 0.5 0.5 0.2 0.3", "1 0.3 0.7 0.1 0.2", "0 0.8 0.2 0.15 0.25"]
        text_content = "\n".join(annotations)

        save_text(filepath, text_content, "annotations")

        # Verify content line by line
        with open(filepath) as f:
            lines = f.read().splitlines()
        assert lines == annotations
