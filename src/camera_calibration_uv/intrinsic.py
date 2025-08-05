import cv2
import numpy as np
from .utils import read_yaml

"""
https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html
"""

def create_board(yaml_path: str) -> cv2.aruco.CharucoBoard:
    board_cfg = read_yaml(yaml_path)
    dict_name = getattr(cv2.aruco, board_cfg["aruco_dict"].upper())
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
    board = cv2.aruco.CharucoBoard(
        size=tuple(board_cfg['size']),
        markerLength=board_cfg['marker_length'],
        squareLength=board_cfg['square_length'],
        dictionary=aruco_dict
    )
    return board

def calibrate_intrinsic(args):
    board = create_board(args.board)
    detector = cv2.aruco.CharucoDetector(board=board)

    input_vid = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.video, False))
    if not input_vid.isOpened(): 
        raise ValueError('Failed to open camera')
    
    all_charuco_corners, all_charuco_ids = [], []
    all_img_points, all_obj_points = [], []

    frame_idx = -1
    while True: 
        frame_idx += 1
        ret, frame = input_vid.read() 
        if not ret: 
            break
        if (frame_idx % args.step_size) != 0: 
            continue
        image_cpy = np.copy(frame)

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image_cpy)
        if not (marker_ids is None) and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_cpy, marker_corners)
        if not (charuco_ids is None) and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(image_cpy, charuco_corners, charuco_ids)
            if len(charuco_ids) > 3: 


        # print("charuco_corners: ", charuco_corners
        #       , "charuco_ids: ", charuco_ids
        #       , "marker_corners: ", marker_corners
        #       , "markers_ids: ", markers_ids)
        
        # if len(charuco_corners) > 3: 
        #     obj_points, img_points = board.matchImagePoints(
        #         charuco_corners, charuco_ids, marker_corners, markers_ids
        #     )
        #     print("Matched points")

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # input_vid.release() 
    # cv2.destroyAllWindows() 