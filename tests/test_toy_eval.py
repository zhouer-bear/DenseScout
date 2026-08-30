from densescout.metrics import recall_from_centers


def test_toy_recall_square_cell():
    records = [{'image_id':'a','width':640,'height':640,'objects':[{'bbox_xyxy':[30,30,34,34]},{'bbox_xyxy':[200,200,220,220]}]}]
    m = recall_from_centers(records, {'a': [(32,32)]})
    assert m['num_gt'] == 2
    assert m['num_hit'] == 1
    assert m['recall'] == 0.5
