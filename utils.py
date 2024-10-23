import cv2
import numpy as np


def get_tag_co_ords(frame, tag_id, detector):
  frame = cv2.resize(frame, (612, 612))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  detections = detector.detect(gray)

  for detection in detections:
      if detection.getId() == tag_id:
        return detection
  return None


def detect_orientation(top_left_co_ords, top_right_co_ords):
  delta_x = top_right_co_ords[0] - top_left_co_ords[0]
  delta_y = top_right_co_ords[1] - top_left_co_ords[1]
  return np.arctan2(delta_y, delta_x) * 180.0 / np.pi


def find_robot_destination_co_ords(post_co_ord, ball_co_ord, distance=5):
  coord1 = np.array(post_co_ord, dtype=np.float32)
  coord2 = np.array(ball_co_ord, dtype=np.float32)

  direction_vector = coord2 - coord1
  direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

  extended_point = coord2 + direction_vector_normalized * distance
  return tuple(extended_point)