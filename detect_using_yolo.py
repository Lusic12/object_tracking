from ultralytics import YOLO


class YOLOv8:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def detect(self, source_img):
        results = self.model.predict(source_img, verbose=False)[0]
        bboxes = results.boxes.xywh.cpu().numpy()
        bboxes[:, :2] = bboxes[:, :2] - (bboxes[:, 2:] / 2)
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        return bboxes, scores, class_ids

