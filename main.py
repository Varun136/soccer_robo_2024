import cv2
import math
from robotpy_apriltag import AprilTagDetector
from utils import get_tag_co_ords, find_robot_destination_co_ords
from ball_detection import detect_ball_co_ords


ROBOT_TAG_CLASS = 0
OPP_POST_TAG_CLASS = 1
TAG_FAMILY = "tag36h11"


detector = AprilTagDetector()
detector.addFamily(TAG_FAMILY)


def main(video_source):
  cap = cv2.VideoCapture(video_source)

  opp_post_co_ords = None
  robot_co_ords = None

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Detect opp post co-ordinates if not already detected.
    if not opp_post_detection:
        opp_post_detection = get_tag_co_ords(frame, OPP_POST_TAG_CLASS, detector)
        while not opp_post_co_ords:
            ret, frame = cap.read()
            opp_post_detection = get_tag_co_ords(frame, OPP_POST_TAG_CLASS, detector)
        opp_post_co_ords = opp_post_detection.center
        print(f"Detected opp post at {opp_post_co_ords}")

    # Track and mark robot co-ordinates
    robot_detection = get_tag_co_ords(frame, ROBOT_TAG_CLASS, detector)
    while not robot_detection:
      print("Robot not detected, Retrying..")
      ret, frame = cap.read()
      robot_detection = get_tag_co_ords(frame, ROBOT_TAG_CLASS, detector)
    robot_co_ords = robot_detection.center
    print(f"Detected robot at {robot_co_ords}")

    ball_co_ords = []
    while not ball_co_ords:
      ball_co_ords, frame = detect_ball_co_ords(frame)


    ball_distances = {}
    for ball_id, ball_co_ord in ball_co_ords.items():
      ball_distance = math.dist(opp_post_co_ords, ball_co_ord)
      ball_distances[ball_distance] = ball_id

    # Select and mark ball.
    selected_ball_id = ball_distances[min(ball_distances.keys())]
    selected_ball_co_ord = ball_co_ords[selected_ball_id]
    print(f"Selected ball at {selected_ball_co_ord}")

    # Choose and mark robo destination
    robot_destination_co_ord = find_robot_destination_co_ords(opp_post_co_ords, selected_ball_co_ord)
    print(f"New robot destination at {robot_destination_co_ord}")

    break
  

if __name__ == "__main__":
  main(0)