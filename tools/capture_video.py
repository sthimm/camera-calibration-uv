import argparse
import os
import cv2

def main(args): 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        raise ValueError('Failed to open camera')
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))

    num_frames = 0 
    max_frames = args.fps * args.duration if args.duration is not None else float('inf')

    while cap.isOpened() and num_frames < max_frames: 
        ret, frame = cap.read() 
        if not ret: 
            print('Failed to capture frame')
            break

        writer.write(frame)
        cv2.imshow('Camera', cv2.flip(frame, 1))
        num_frames += 1 

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            print('Stopped recording')
            break

    cap.release() 
    writer.release() 
    cv2.destroyAllWindows() 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--output-path', type=str, default='output/output.mp4', help='Path to output dir')
    parser.add_argument('--fps', type=int, default=20, help='Acquisition framerate')
    parser.add_argument('--duration', type=int, default=None, help='Optional video length (seconds)')
    args = parser.parse_args() 

    main(args)