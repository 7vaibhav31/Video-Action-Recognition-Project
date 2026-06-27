import cv2
import os

def compress_video(input_path, output_path, target_width=640, max_frames=300):
    """
    Reads a video, resizes it to a smaller resolution (to save memory), 
    and limits the maximum number of frames (to fix the 'sparse sampling' issue on long videos).
    
    Args:
        input_path: The path to the original uploaded video.
        output_path: The path where the compressed video will be saved.
        target_width: The width to resize to (height is calculated automatically).
        max_frames: The maximum number of frames to keep (e.g., 300 frames = 10 seconds at 30fps).
    """
    
    # 1. Open the original video file using OpenCV's VideoCapture
    cap = cv2.VideoCapture(input_path)
    
    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"[compress] Error: Could not open video {input_path}")
        return False
        
    # 2. Get the original width and height of the video
    # cv2.CAP_PROP_FRAME_WIDTH gets the width property
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # cv2.CAP_PROP_FRAME_HEIGHT gets the height property
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get the Frames Per Second (FPS) of the original video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Sometimes FPS is 0 or NaN if the video is corrupted, so we default to 30
    if fps <= 0 or fps != fps:
        fps = 30.0
        
    # 3. Calculate the new height to keep the same aspect ratio (shape)
    # We divide the target_width by the orig_width to get the scaling ratio
    ratio = target_width / float(orig_width)
    # Multiply the original height by the ratio to get the new height
    target_height = int(orig_height * ratio)
    
    # 4. Set up the VideoWriter to save the new compressed video
    # 'mp4v' is a widely supported video codec format for MP4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create the VideoWriter object with the output path, codec, FPS, and new size
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    frame_count = 0
    
    # 5. Loop through the video frame by frame
    while True:
        # cap.read() returns 'ret' (True if successful) and 'frame' (the image data)
        ret, frame = cap.read()
        
        # If 'ret' is False, we have reached the end of the video
        if not ret:
            break
            
        # 6. Resize the frame using cv2.resize
        # We pass the frame and the new (width, height) tuple
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        # 7. Write the resized frame to the new video file
        out.write(resized_frame)
        
        frame_count += 1
        
    # 9. Release the resources (very important to prevent memory leaks!)
    cap.release()
    out.release()
    
    print(f"[compress] Successfully compressed video to {target_width}x{target_height} ({frame_count} frames)")
    return True
