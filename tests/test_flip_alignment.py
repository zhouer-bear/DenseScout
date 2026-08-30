import numpy as np
from densescout.transforms import flip_image_and_record
from densescout.metrics import bbox_center_xyxy


def test_horizontal_flip_bbox_center_alignment():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    rec = {'width': 200, 'height': 100, 'objects': [{'bbox_xyxy': [10, 20, 30, 40]}]}
    _, out = flip_image_and_record(img, rec, horizontal=True)
    assert out['objects'][0]['bbox_xyxy'] == [170.0, 20, 190.0, 40]
    assert bbox_center_xyxy(out['objects'][0]['bbox_xyxy']) == (180.0, 30.0)


def test_vertical_flip_bbox_center_alignment():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    rec = {'width': 200, 'height': 100, 'objects': [{'bbox_xyxy': [10, 20, 30, 40]}]}
    _, out = flip_image_and_record(img, rec, vertical=True)
    assert out['objects'][0]['bbox_xyxy'] == [10, 60.0, 30, 80.0]
    assert bbox_center_xyxy(out['objects'][0]['bbox_xyxy']) == (20.0, 70.0)
