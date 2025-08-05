import cv2

def calibrate_intrinsic(args):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened(): 
        raise ValueError('Failed to open camera')
    
    frame_idx = -1
    while True: 
        frame_idx += 1
        ret, frame = cap.read() 
        if not ret: 
            break
        if (frame_idx % args.step_size) != 0: 
            continue

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release() 
    cv2.destroyAllWindows() 