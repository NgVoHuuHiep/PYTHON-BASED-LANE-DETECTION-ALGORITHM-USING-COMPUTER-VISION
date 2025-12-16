'''
Nguyen Vo Huu Hiep
What I have done:
- Lane detection (kinda work for little curved lane)
- Save output video to corresponding directory
- Offset of the vehicle (make a beep when occur)
- Check for anomalies (abnormal image will be saved to corresponding directory)
- Display FPS

Note: Start applying Machine Learning on Image dataset, from the video to Steering output (Simulation application)
Two approachs: From input to lane detection, from lane detection to steering output
'''
import cv2
import numpy as np
import os
import winsound as ws
import threading as th

# Straight lane videos:
vid1_1 = "data/straight/america_straight.mkv"
vid1_2 = "data/straight/swiss_straight.mkv" # This video needed fixing. Solution: change HoughLine threshold to 150

# Curve lane videos:
vid2_1 = "data/curve/america_curved.mkv"
vid2_2 = "data/curve/highway_curve.mp4"
vid2_3 = "data/curve/monaco_curved.mkv" # This video needed fixing

# Mix lane videos:
vid3_1 = "data/mix/vancouver_mix.mkv"
vid3_2 = "data/mix/LaneVideo.mp4"

# Japan test videos:
japan1 = "data/japan/japan_normal.mp4"
japan2 = "data/japan/japan_bright_tunnel.mp4"
japan3 = "data/japan/japan_dark_tunnel.mp4"
japan4 = "data/japan/japan_night_bridge.mp4"
japan5 = "data/japan/japan_night_curve.mp4"
japan6 = "data/japan/japan_difficult.mp4"

# Path for input data
in_file_name = vid2_1
out_file_name = os.path.join("results/mix", os.path.splitext(os.path.basename(in_file_name))[0] + "_result.mp4")

# Some setups for later uses
vid_write = False   # Set to "True" to download the output video to the corresponding directory
check_anomalies = False  # Set to "True" to load the anomalies image to corresponding folder
warning_sound = False   # Set to "True" to make a beep when departure occurs

# Detect edges: convert to grayscale, apply Gaussian blur, apply Canny algorithm
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    ret = cv2.Canny(gray, 50, 150)
    return ret


# Returns an image that has 255 (white) for only required area, acting as a "mask". To apply it, the image is "ANDed" (multiply) with the mask
def returnAoi(image):
    height = image.shape[0]
    width = image.shape[1]
    # Adjust Region of Interest
    polygon = np.array([
        [
            (int(width * 0.10), int(height)),
            (int(width * 0.30), int(height * 0.70)),
            (int(width * 0.70), int(height * 0.70)),
            (int(width * 0.90), int(height))
        ]
    ])
    ret = np.zeros_like(image)
    cv2.fillPoly(ret, polygon, 255)
    return ret


# Returns an image that has all the lines passed
def displayLines(image, lines, colour=(255, 255, 255)): # The color is for Detected lines window
    line_image = np.zeros_like(image)
    if (lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line.astype(int)
            if x1 == x2:
                continue  # Skip vertical lines to avoid division by zero

            slope = (y1 - y2) / (x1 - x2)
            if (abs(slope) < 0.2) or (abs(slope) > 20):
                continue  # Skip nearly horizontal or overly steep lines

            cv2.line(line_image, (x1, y1), (x2, y2), colour, 10)
    return line_image


# Displays a mask of area between passed lane boundaries. Used for overlaying on original image
def displayLane(image, l_line, r_line, color=(255, 0, 0)):
    roi_image = np.zeros_like(image)
    if (l_line == (0, 0, 0, 0)).all():
        return roi_image

    if (r_line == (0, 0, 0, 0)).all():
        return roi_image

    polygon = np.zeros((1, 4, 2))

    polygon[0][0] = (l_line[0], l_line[1])
    polygon[0][1] = (l_line[2], l_line[3])
    polygon[0][2] = (r_line[2], r_line[3])
    polygon[0][3] = (r_line[0], r_line[1])

    polygon = polygon.astype(int)
    cv2.fillPoly(roi_image, polygon, color)
    return roi_image


def getAverageLines(image, lines):
    left_fit = []
    right_fit = []

    max_length_for_dashed = 150

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x2 - x1) == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if length > max_length_for_dashed:
            continue

        if slope < -0.5:
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))

    if len(left_fit) == 0 or len(right_fit) == 0:
        return np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])

    left_avg_slope = np.mean([fit[0] for fit in left_fit])
    left_avg_intercept = np.mean([fit[1] for fit in left_fit])

    right_avg_slope = np.mean([fit[0] for fit in right_fit])
    right_avg_intercept = np.mean([fit[1] for fit in right_fit])

    # 1. Calculate intersection point (x, y) of left and right lines
    if (left_avg_slope - right_avg_slope) == 0:
        intersection_x = image.shape[1] // 2
    else:
        intersection_x = int((right_avg_intercept - left_avg_intercept) / (left_avg_slope - right_avg_slope))

    intersection_y = int(left_avg_slope * intersection_x + left_avg_intercept)

    # 2. Set y1 (bottom of image), and y2 (higher point, but not too high)
    y1 = image.shape[0]
    y2 = max(intersection_y, int(image.shape[0] * 0.7))  # don't go above 60% height

    x1_left = int((y1 - left_avg_intercept) / left_avg_slope) if left_avg_slope != 0 else 0
    x2_left = int((y2 - left_avg_intercept) / left_avg_slope) if left_avg_slope != 0 else 0

    x1_right = int((y1 - right_avg_intercept) / right_avg_slope) if right_avg_slope != 0 else image.shape[1]
    x2_right = int((y2 - right_avg_intercept) / right_avg_slope) if right_avg_slope != 0 else image.shape[1]

    left_line = np.array([x1_left, y1, x2_left, y2])
    right_line = np.array([x1_right, y1, x2_right, y2])

    return right_line, left_line


# Returns the coordinates of line from slope and intercept form
def getCoords(image, para):
    if (para == (0, 0)).all():
        return np.array((0, 0, 0, 0))
    m = para[0]
    b = para[1]
    height = image.shape[0]     # 1920
    width = image.shape[1]      # 1080
    y1 = height
    y2 = 2 * y1 / 3
    x1 = (y1 - b) / m
    x2 = (y2 - b) / m
    if (x1 < 0):
        x1 = 0
        y1 = (m * x1) + b
    if (x2 < 0):
        x2 = 0
        y2 = (m * x2) + b

    if (x1 > width):
        x1 = width
        y1 = (m * x1) + b
    if (x2 > width):
        x2 = width
        y2 = (m * x2) + b
    return np.maximum(np.minimum(np.array((x1, y1, x2, y2)), 10000 * np.ones(4)), -10000 * np.ones(4))


def beep():
    ws.Beep(frequency=5000, duration=2)

# Anomalies check
anomaly_dir = "results/anomalies"
frame_count = 0
prev_l_line = None

video = cv2.VideoCapture(in_file_name)

# Initialise video writer and video reader
if (vid_write):
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1200, 700)) # (frame_width, frame_height))
    print("\nGenerating video output...\n")

# Ratio for calculating exponential moving average (EMA) - increasing this value increases smoothness, but increases delay
sum_ratio = 0.8
# Variables storing EMA for lane boundary lines
l_line_sum, r_line_sum = None, None


while (video.isOpened()):
    start_time = cv2.getTickCount()
    # Get image from capture
    _, image = video.read()
    if image is None:
        break

    # Processing stages : Resize -> Edge detect(canny) -> Apply roi mask -> Detect lines(Hough)
    image = cv2.resize(image, (1200, 700))

    # image = image.copy()
    canny_image = canny(image)
    aoi = returnAoi(canny_image)
    interest = canny_image & aoi
    lines = cv2.HoughLinesP(interest, 1, np.pi / 180, 100, minLineLength = 20, maxLineGap = 5)

    # If no lines are detected, set right and left lines to (0,0,0,0)
    if lines is None:
        orig_lines_img = np.zeros_like(image)
        right_line = np.array((0, 0, 0, 0))
        left_line = right_line
    else:
        orig_lines_img = displayLines(image, lines.reshape((-1, 4)))
        right_line, left_line = getAverageLines(image, lines)

    # If EMA is being calculated for first time, set as detected lines
    if (l_line_sum is None):
        l_line_sum = left_line
    if (r_line_sum is None):
        r_line_sum = right_line
    if (l_line_sum == (0, 0, 0, 0)).all():
        l_line_sum = left_line
    if (r_line_sum == (0, 0, 0, 0)).all():
        r_line_sum = right_line

    # Calculate next EMA
    smoothing_enabled = True   # Set to "False" for debugging flickering
    if smoothing_enabled:
        if (left_line != (0, 0, 0, 0)).any():
            l_line_sum = sum_ratio * l_line_sum + (1 - sum_ratio) * left_line
        if (right_line != (0, 0, 0, 0)).any():
            r_line_sum = sum_ratio * r_line_sum + (1 - sum_ratio) * right_line
    else:
        l_line_sum = left_line
        r_line_sum = right_line

    # Make images for lines and roi, and blend them together
    # Right and Left lane markings
    r_lines_img = displayLines(image, r_line_sum.reshape((1, 4)), (255, 0, 0)) # Blue
    l_lines_img = displayLines(image, l_line_sum.reshape((1, 4)), (0, 0, 255)) # Red
    lines_img = cv2.addWeighted(r_lines_img, 2, l_lines_img, 2, 1)
    # Overlay lane markings to the frame
    roi_img = displayLane(image, l_line_sum, r_line_sum, (0, 255, 0)) # Green lane in between 2 lane markings
    blend_img = cv2.addWeighted(roi_img, 0.7, image, 0.8, 1)
    blend_img = cv2.addWeighted(lines_img, 1, blend_img, 0.8, 1)
    # Region of Interested overlay
    aoi_img = np.zeros_like(image)
    aoi_img[:, :, 2] = aoi
    blend_img = cv2.addWeighted(aoi_img, 0.3, blend_img, 1, 1)

    # Lane Departure Warning
    image_center = int(image.shape[1] / 2) # x - axis
    image_center_y = int(image.shape[0] - 10) # y - axis
    cv2.circle(blend_img, (image_center, image_center_y), radius=5, color=(0, 0, 255), thickness=-1)


    if (l_line_sum != (0, 0, 0, 0)).any() and (r_line_sum != (0, 0, 0, 0)).any():
        lane_center = (l_line_sum[0] + r_line_sum[0]) / 2
        offset = image_center - lane_center  # Positive if car is to the right, negative if to the left

        # Set the length from midpoint to the offset edge
        offset_threshold = 100

        if offset > offset_threshold:
            cv2.putText(blend_img, "Offset right", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if warning_sound:
                th.Thread(target=beep).start()
        elif offset < -offset_threshold:
            cv2.putText(blend_img, "Offset left", (990, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if warning_sound:
                th.Thread(target=beep).start()
        else:
            cv2.putText(blend_img, "Inside lane", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Compute FPS
    end_time = cv2.getTickCount()
    time_taken = (end_time - start_time) / cv2.getTickFrequency()
    fps = 1.0 / time_taken
    cv2.putText(blend_img, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Check anomalies
    if (check_anomalies):
        frame_count += 1
        anomaly = False
        reason = ""

        # Error: No lanes
        if (l_line_sum == (0, 0, 0, 0)).all() or (r_line_sum == (0, 0, 0, 0)).all():
            anomaly = True
            reason = "no_detected_lane"

        # Error: Sudden jump in detection
        jump_threshold = 50  # <-- Tune this number for sudden jump error
        if prev_l_line is not None:
            jump = np.linalg.norm(l_line_sum - prev_l_line) > jump_threshold
            if jump:
                anomaly = True
                reason = "sudden_jump"
        prev_l_line = l_line_sum.copy()

        # Save anomaly frame
        if anomaly:
            display_frame = blend_img.copy()
            cv2.putText(display_frame, f"Anomaly: {reason}", (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            basename = os.path.splitext(os.path.basename(in_file_name))[0]
            filename = os.path.join(anomaly_dir, f"frame_{frame_count:05d}_{reason}_{basename}.jpg")
            cv2.imwrite(filename, display_frame)

    # Write the frame to video file
    if (vid_write):
        out.write(blend_img)

    # Display the frames and all lines detected
    cv2.imshow("Lane detection", blend_img)
    cv2.imshow("Detected lines", orig_lines_img)

    if cv2.waitKey(1) == ord('q'):
        break

# Close all opened writers, readers and windows
if (vid_write):
    out.release()
    print('Video output is generated.\nVideo is stored in "results" folder.\n')
video.release()
cv2.destroyAllWindows()