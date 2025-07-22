from typing import List

import yaml


def xywh2xyxy(xywh, scale=(1, 1)):
    """Converts bounding box coordinates from center, width, and height format to
    top-left and bottom-right format.
    Args:
        xywh (list): List of bounding box coordinates in center, width, and height format.
        scale (tuple): Tuple of image width and height.
    Returns:
        list: List of bounding box coordinates in top-left and bottom-right format.
    """

    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = x1 + xywh[2]
    y2 = y1 + xywh[3]
    return [x1 * scale[0], y1 * scale[1], x2 * scale[0], y2 * scale[1]]


def find_missing_number(nums: List[int]) -> int:
    """Finds the smallest missing positive integer in a sorted list of integers.
    Args:
        nums (List[int]): A sorted list of integers.

    Returns:
        int: The smallest missing positive integer in the list.
    """
    if not nums:  # Handle empty list
        return 0

    nums.sort()
    if nums[0] != 0:
        return 0
    for i in range(len(nums) - 1):
        if nums[i + 1] - nums[i] > 1:
            return nums[i] + 1
    return nums[-1] + 1


def save_text(filepath, text, file_type):
    if file_type == "annotations":
        with open(filepath, "w") as f:
            f.write(text)
    elif file_type == "classlist":
        with open(filepath, "w") as file:
            yaml.dump(text, file, default_flow_style=False)
    else:
        raise ValueError(f"Invalid file_type: {file_type}")
