import cv2

def background_subtraction(video_path):
    # Create the background subtractor object
    backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    # Open the video
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print("Error opening video file.")
        return

    frame_count = 0  # Initialize a frame counter

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1  # Increment the frame counter

        # Blur frame to reduce noise and improve background subtraction
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Apply the background subtractor to get the foreground mask
        fgMask = backSub.apply(blurred_frame)

        # Convert fgMask to a 3-channel image so it can be concatenated with the color frame
        fgMask_3channel = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)

        # Generate the result frame by combining the original frame and the mask
        result_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

        if frame_count > 4:  # Only display the results after the 4th frame
            # Display the original frame, the mask, and the result side by side
            combined_frame = cv2.hconcat([frame, fgMask_3channel, result_frame])
            cv2.imshow('Background Subtraction', combined_frame)

        keyboard = cv2.waitKey(30)
        if keyboard == ord('q') or keyboard == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

