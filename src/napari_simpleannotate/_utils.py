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
