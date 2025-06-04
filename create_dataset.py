from datasets import load_dataset

def coco_to_xyxy(coco_bbox):
    x, y, width, height = coco_bbox
    x1, y1 = x, y
    x2, y2 = x + width, y + height
    return [x1, y1, x2, y2]

def coco_cat_to_name(coco_cat):
    cat_to_name = {
        0: '__background__',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'backpack',
        26: 'umbrella',
        27: 'handbag',
        28: 'tie',
        29: 'suitcase',
        30: 'frisbee',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket',
        40: 'bottle',
        41: 'wine glass',
        42: 'cup',
        43: 'fork',
        44: 'knife',
        45: 'spoon',
        46: 'bowl',
        47: 'banana',
        48: 'apple',
        49: 'sandwich',
        50: 'orange',
        51: 'broccoli',
        52: 'carrot',
        53: 'hot dog',
        54: 'pizza',
        55: 'donut',
        56: 'cake',
        57: 'chair',
        58: 'couch',
        59: 'potted plant',
        60: 'bed',
        61: 'dining table',
        62: 'toilet',
        63: 'tv',
        64: 'laptop',
        65: 'mouse',
        66: 'remote',
        67: 'keyboard',
        68: 'cell phone',
        69: 'microwave',
        70: 'oven',
        71: 'toaster',
        72: 'sink',
        73: 'refrigerator',
        74: 'book',
        75: 'clock',
        76: 'vase',
        77: 'scissors',
        78: 'teddy bear',
        79: 'hair drier',
        80: 'toothbrush'
    }
    return cat_to_name[int(coco_cat)+1]

def convert_to_detection_string(bboxs, image_width, image_height, cats):
    def format_location(value, max_value):
        return f"<loc{int(round(value * 1024 / max_value)):04}>"

    detection_strings = []
    for bbox, cat in zip(bboxs, cats):
        x1, y1, x2, y2 = coco_to_xyxy(bbox)
        name = coco_cat_to_name(cat)
        locs = [
            format_location(y1, image_height),
            format_location(x1, image_width),
            format_location(y2, image_height),
            format_location(x2, image_width),
        ]
        detection_string = "".join(locs) + f" {name}"
        detection_strings.append(detection_string)

    return " ; ".join(detection_strings)


def format_objects(example):
    height = example["height"]
    width = example["width"]
    bboxs = example["objects"]["bbox"]
    cats = example["objects"]["category"]
    formatted_objects = convert_to_detection_string(bboxs, width, height, cats)
    return {"label_for_paligemma": formatted_objects}


if __name__ == "__main__":
    # load the dataset
    dataset_id = "detection-datasets/coco"
    print(f"[INFO] loading {dataset_id} from hub...")
    dataset = load_dataset(dataset_id)

    # modify the coco bbox format
    dataset["train"] = dataset["train"].map(format_objects)
    dataset["val"] = dataset["val"].map(format_objects)

    # push to hub
    dataset.push_to_hub("savoji/coco-paligemma")
