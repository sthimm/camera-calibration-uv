import cv2 
import json 
import os
import yaml 
import argparse

def create_board(yaml_path: str) -> cv2.aruco.CharucoBoard:
    with open(yaml_path, 'r') as file:
        board_cfg = yaml.safe_load(file)
    aruco_dict_name = getattr(cv2.aruco, board_cfg['aruco_dict'].upper())
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    board = cv2.aruco.CharucoBoard(
        size=tuple(board_cfg['size']),
        markerLength=board_cfg['marker_length'],
        squareLength=board_cfg['square_length'],
        dictionary=aruco_dict
    )
    return board

def calibrate_intrinsic(args):
    """
    References:  
    https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html
    https://github.com/opencv/opencv/blob/4.x/samples/python/aruco_detect_board_charuco.py#L72
    https://github.com/opencv/opencv/blob/4.x/samples/python/calibrate.py
    """
    board = create_board(args.board)
    detector = cv2.aruco.CharucoDetector(board=board)

    input_vid = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.video, False))
    if not input_vid.isOpened(): 
        raise ValueError('Failed to open video')
    
    img_points, obj_points = [], []
    frame_idx = -1
    image_size = None

    while True:
        frame_idx += 1
        ret, frame = input_vid.read() 
        if not ret: 
            break
        if (frame_idx % args.step_size) != 0: 
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1] # (w, h)

        # Detect charuco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if not (marker_ids is None) and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners)
        if not (charuco_ids is None) and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            if len(charuco_ids) >= args.min_corners: 
                try: 
                    frame_obj_points, frame_img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                    img_points.append(frame_img_points)
                    obj_points.append(frame_obj_points)
                    print(f'Frame {frame_idx}: Found {len(charuco_ids)} corners')
                except cv2.error as e: 
                    print(f'Point matching failed: {e}')

        cv2.imshow('Camera', frame)
        key = cv2.waitKey(args.wait_time) & 0xFF
        if key == ord('q'): 
            break

    input_vid.release() 
    cv2.destroyAllWindows() 

    num_points = sum(len(p) for p in img_points)
    print(f'Using {len(img_points)} frames with {num_points} matched points for calibration')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, 
        img_points, 
        image_size,
        cameraMatrix=None,
        distCoeffs=None,
        criteria=criteria,
    )

    print("Re-projection error:", rms)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients: ", dist_coefs.ravel())

    calib_results = {
        'reprojection_error': rms,
        'camera_matrix': camera_matrix.ravel().tolist(),
        'dist_coefs': dist_coefs.ravel().tolist()
    }

    assert args.output_path.endswith('.json'), 'Output path must end with .yaml'
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as file:
        json.dump(calib_results, file)

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--video', type=str, default='output/output.mp4', help='Path to video file')
    parser.add_argument('--output-path', type=str, default='output/intrinsics.json', help='Path to output dir')
    parser.add_argument('--board', type=str, default='example_boards/charuco_5x7.yaml', help='YAML File describing charuco board')
    parser.add_argument('--step-size', type=int, default=1, help='Use every n-th frame of video')
    parser.add_argument('--min-corners', type=int, default=4, help='Minimum number of detected corners')
    parser.add_argument('--wait-time', type=int, default=1, help='Time to wait between frames in milliseconds')

    args = parser.parse_args() 
    calibrate_intrinsic(args)