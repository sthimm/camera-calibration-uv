from dataclasses import dataclass
from typing import Tuple
import yaml
import cv2

@dataclass
class CharucoBoard:
    """ https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html
    """
    size: Tuple[int, int] = (5, 7) 
    aruco_dict: str = 'DICT_4X4_50'

    square_length: float = 0.112
    marker_length: float = 0.084

    min_rows: int = 2 
    min_points: int = 10 

    _board: cv2.aruco.CharucoBoard = None

    def __post_init__(self):
        dict_name = getattr(cv2.aruco, f'DICT_{self.aruco_dict.upper()}')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)

    @staticmethod
    def from_yaml(yaml_path: str) -> 'CharucoBoard': 
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return CharucoBoard(**data)

    @property 
    def board(self) -> cv2.aruco.CharucoBoard:
        if self._board is None:
            self._board = cv2.aruco.CharucoBoard(
                size=self.size,
                markerLength=self.marker_length,
                squareLength=self.square_length,
                dictionary=self.aruco_dict
            )
        return self._board
    
    @property
    def detector(self) -> cv2.aruco.CharucoDetector: 
        return cv2.aruco.CharucoDetector(board=self.board)