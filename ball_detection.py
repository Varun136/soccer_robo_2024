import cv2
import numpy as np

# Adjust accoording to lighting.
YELLOW_LOWER_LIM = [15, 200, 200]
YELLOW_UPPER_LIM = [35, 255, 255]

MIN_RADIUS = 35
MARK_RESULT = False

def detect_ball_co_ords(frame):
    """
    hue - color type: red, green, blue (15-35 -> yellow)
    saturation - purity of the color
    value - brightness of the color (Adjust the brightness)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array(YELLOW_LOWER_LIM)
    upper_yellow = np.array(YELLOW_UPPER_LIM)

    # Masks out all other colors and find contours with yellow color.
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)
    
    ball_co_ords = []

    for contour in contours:
        # Filter out small contours.
        area = cv2.contourArea(contour)
        if  area< 100:
            continue
        
        # Check the minimum radius of the cirlce.
        _, radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        
        # Validate the circularity.
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        if circularity < 0.8 and radius < MIN_RADIUS:
            continue
        
        # Find the center co-ords.
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            ball_co_ords.append((cx, cy))

            if MARK_RESULT:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
    return ball_co_ords, frame



# Test.
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Unable to open camera")
    
    # Get the width and height of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (for .mp4 format)
    out = cv2.VideoWriter('output_with_detection.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        co_ords, frame = detect_ball_co_ords(frame)
        print(co_ords)     
        out.write(frame)
        cv2.imshow('Ball Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


