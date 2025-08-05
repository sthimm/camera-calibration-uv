import argparse
import os
import cv2

def main(args): 
    cam = cv2.VideoCapture(0)
    if not cam.isOpened(): 
        raise ValueError("Failed to open camera")
    
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))

    num_frames = 0 
    max_frames = args.fps * args.duration if args.duration is not None else float('inf')

    while cam.isOpened() and num_frames < max_frames: 
        ret, frame = cam.read() 
        if not ret: 
            print('Failed to capture frame')
            break

        writer.write(frame)
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'): 
            break

    cam.release() 
    writer.release() 
    cv2.destroyAllWindows() 
    print(f'Saved video file to {args.output_path}')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--output-path', type=str, default='output/output.mp4', help='Path to output dir')
    parser.add_argument('--fps', type=int, default=20, help='Acquisition framerate')
    parser.add_argument('--duration', type=int, default=None, help='Optional video length (seconds)')
    args = parser.parse_args() 

    main(args)