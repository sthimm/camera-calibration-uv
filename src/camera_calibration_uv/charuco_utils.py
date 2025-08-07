import cv2 
import numpy as np 
import yaml

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

def draw_board(
    board: cv2.aruco.CharucoBoard, 
    size_px: tuple, 
    margin_px: int = 0,
    border_bits: int = 1
) -> np.ndarray:
    img = board.generateImage(
        outSize=size_px,
        marginSize=margin_px,
        borderBits=border_bits
    )
    return img
