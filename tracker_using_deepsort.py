from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import numpy as np
import cv2
from detection import YOLOv8
import os
import torch
import time

# Kiểm tra xem CUDA có sẵn không
if torch.cuda.is_available():
    # Sử dụng thiết bị GPU đầu tiên nếu có
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    # Nếu không, sử dụng CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

class DeepSORT:
    def __init__(self, model_path='resources/networks/mars-small128.pb', max_cosine_distance=0.7, nn_budget=None, classes=['objects']):
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

        key_list = []
        val_list = []
        for ID, class_name in enumerate(classes):
            key_list.append(ID)
            val_list.append(class_name)
        self.key_list = key_list
        self.val_list = val_list

    def tracking(self, origin_frame, bboxes, scores, class_ids):
        features = self.encoder(origin_frame, bboxes)

        detections = [Detection(bbox, score, class_id, feature)
                      for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features)]

        self.tracker.predict()
        self.tracker.update(detections)

        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 10:
                continue
            bbox = track.to_tlbr()
            class_id = track.get_class()
            conf_score = track.get_conf_score()
            tracking_id = track.track_id
            start_time =time.time()
            tracked_bboxes.append(bbox.tolist() + [class_id, conf_score, tracking_id,start_time])


        tracked_bboxes = np.array(tracked_bboxes)

        return tracked_bboxes

def draw_detection(img, bboxes, scores, class_ids, ids, classes=['objects'], id_status=None, start_times=None, mask_alpha=0.3):
    if id_status is None:
        id_status = {}
    if start_times is None:
        start_times = {}

    height, width = img.shape[:2]
    np.random.seed(0)
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(classes), 3))

    mask_img = img.copy()
    det_img = img.copy()

    size = min([height, width]) * 0.0006
    text_thickness = int(min([height, width]) * 0.001)

    for bbox, score, class_id, id_, start_time in zip(bboxes, scores, class_ids, ids, start_times):
        color = colors[class_id]

        x1, y1, x2, y2 = bbox.astype(int)

        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = classes[class_id]
        if id_ in id_status:
            id_status_str = id_status[id_]
        else:
            id_status_str = "Unknown"
        if id_ in start_times:
            elapsed_time = int(time.time() - start_times[id_])
        else:
            elapsed_time = 0
        caption = f'{label} {int(score * 100)}% ID: {id_} Status: {id_status_str} Time: {elapsed_time}s'  # Display elapsed time
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

def tracker(frame, detector=YOLOv8('best.pt'), tracker=DeepSORT()):
    tracked_ids = np.array([], dtype=np.int32)
    id_status = {}  # Biến để lưu trạng thái của mỗi ID
    start_times = {}  # Biến để lưu thời gian bắt đầu của mỗi ID

    while True:
        # Phát hiện đối tượng trên khung hình
        detector_results = detector.detect(frame)
        bboxes, scores, class_ids = detector_results

        # Dự đoán đối tượng sử dụng trình theo dõi
        tracker_pred = tracker.tracking(origin_frame=frame, bboxes=bboxes, scores=scores, class_ids=class_ids)

        if tracker_pred.size > 0:
            # Lấy các thông tin từ dự đoán của trình theo dõi
            bboxes = tracker_pred[:, :4]
            class_ids = tracker_pred[:, 4].astype(int)
            conf_scores = tracker_pred[:, 5]
            tracking_ids = tracker_pred[:, 6].astype(int)
            start_times = tracker_pred[:, 7].astype(int)

            # Tìm ID mới và cũ
            new_ids = np.setdiff1d(tracking_ids, tracked_ids)
            old_ids = np.intersect1d(tracked_ids, tracking_ids)

            # Cập nhật trạng thái của các ID
            for id_ in new_ids:
                id_status[id_] = 'new'
            for id_ in old_ids:
                id_status[id_] = 'old'

            # Cập nhật tracked_ids
            tracked_ids = np.concatenate((tracked_ids, new_ids))

            # Vẽ các hộp giới hạn và nhãn trên khung hình
            result_img = draw_detection(img=frame, bboxes=bboxes, scores=conf_scores,
                                        class_ids=class_ids, ids=tracking_ids, id_status=id_status, start_times=start_times)
        else:
            result_img = frame

        return result_img
