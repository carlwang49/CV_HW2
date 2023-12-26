import cv2
import numpy as np

def preprocessing(video_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=1,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Detect features in the image
    points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    if points is not None:
        # Take the first and only point
        x, y = points[0].ravel()

        # Draw a red cross at the detected point
        cross_color = (0, 0, 255) # Red color in BGR
        cross_size = 10 # Length of the cross arms
        cross_thickness = 4 # Thickness of the cross lines
        cv2.line(frame, (int(x) - cross_size, int(y)), (int(x) + cross_size, int(y)), cross_color, cross_thickness)
        cv2.line(frame, (int(x), int(y) - cross_size), (int(x), int(y) + cross_size), cross_color, cross_thickness)
        cv2.imshow("Frame with Feature", frame)
        cv2.waitKey(3000)
    else:
        x, y = None, None

    # Release video capture object and close all frames
    cap.release()
    cv2.destroyAllWindows()

    # Return the coordinates of the detected point
    return np.array([[x, y]], dtype=np.float32) if x is not None and y is not None else None


def video_tracking(video_path, initial_point):
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Color for the trajectory
    color = (0, 100, 255)  # Highly visible yellow color

    # Take the first frame and convert it to grayscale
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing the tracking line
    mask = np.zeros_like(old_frame)

    # Track the point across the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow to track the point
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, initial_point, None, **lk_params)

        # Select good points
        if p1 is not None and st[0][0] == 1:
            # Draw the trajectory line
            end_point = tuple(p1.ravel().astype(int))
            initial_point_int = tuple(initial_point.ravel().astype(int))
            mask = cv2.line(mask, initial_point_int, end_point, color, 2)
            frame = cv2.circle(frame, end_point, 5, color, -1)
            
            # Update the initial point for the next frame
            initial_point = p1

        # Overlay the tracking line on the video
        img = cv2.add(frame, mask)

        # Display the frame
        cv2.imshow('Video Tracking', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # Exit if ESC is pressed
            break

        # Update the previous frame and points for the next iteration
        old_gray = frame_gray.copy()

    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    cap.release()


