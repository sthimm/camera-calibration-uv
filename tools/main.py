import cv2
import numpy as np
import json
import os
import natsort
import argparse
from tqdm import tqdm
def create_charuco_board(size, aruco_dict, square_length, marker_length):
    board = cv2.aruco.CharucoBoard_create(size[0], size[1], square_length, marker_length, aruco_dict)
    return board

def save_charuco_board_image(board, image_path, image_size=(1920, 1080)):
    image = board.draw(image_size)
    cv2.imwrite(image_path, image)

def calibrate_camera(images_folder, board, square_length, marker_length):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    step = 1
    obj_points = []
    charuco_idss = []# 3D points in real world space
    img_points = []  # 2D points in image plane

    objp = np.zeros((np.prod(board.getChessboardSize()), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board.getChessboardSize()[0], 0:board.getChessboardSize()[1]].T.reshape(-1, 2)
    objp *= square_length

    findImageNames = []
    # for file_name in tqdm(natsort.natsorted(os.listdir(images_folder))[0::2]):
    for file_name in tqdm(natsort.natsorted(os.listdir(images_folder))[0::step]):
        # if file_name.endswith(('.jpg', '.png', '.jpeg')) and (camera_name in file_name):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(images_folder, file_name))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = cv2.aruco.detectMarkers(gray, board.dictionary)
            if len(corners) > 3:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 6:
                    # Convert charuco_ids to a 1D array
                    charuco_ids_1d = charuco_ids.ravel()
                    # Find the indices of the matching charuco_ids in objp
                    # indices = np.where(np.isin(objp[:, 0].astype(int), charuco_ids_1d))
                    # Use boolean indexing to get the selected rows from objp
                    selected_objp = objp[charuco_ids_1d]
                    obj_points.append(selected_objp)
                    charuco_idss.append(charuco_ids)
                    img_points.append(charuco_corners)
                    findImageNames.append(file_name)
    # flags = cv2.CALIB_USE_QR  # USE FOR POTENTIALLY QUICKER CALIBRATION....
    print("starting calibration process.....\n This takes some time, depending on how many images are used.")
    # intrinsicGuess = np.array([[1411.2120405906408, 0.0, 981.3683986197668],
    #                            [0.0, 1411.1279090563007, 635.2864733688431],
    #                            [0.0, 0.0, 1.0]])
    # distCoeffs = np.array([
    #     [
    #         -0.16876852877429152,
    #         0.11413519405810994,
    #         0.0010242310454136932,
    #         0.0001996175693553593,
    #         -0.0036399759297398348
    #     ]
    # ])
    # ret, camera_matrix, distortion_coefficients, _, _ = cv2.aruco.calibrateCameraCharuco(
    #     img_points, charuco_idss, board, gray.shape[::-1], cameraMatrix=intrinsicGuess, distCoeffs=distCoeffs, criteria=criteria, flags=cv2.CALIB_USE_INTRINSIC_GUESS
    # )

    # ret, camera_matrix, distortion_coefficients, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
    #     img_points, charuco_idss, board, gray.shape[::-1], cameraMatrix=intrinsicGuess, distCoeffs=distCoeffs,
    #     criteria=criteria, flags=cv2.CALIB_USE_INTRINSIC_GUESS
    # )
    ret, camera_matrix, distortion_coefficients, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
        img_points, charuco_idss, board, gray.shape[::-1], cameraMatrix=np.zeros((3, 3)), distCoeffs=np.zeros((1, 5)),
        criteria=criteria
    )

    print("finished calibration process...")

    return ret, camera_matrix, distortion_coefficients

def save_calibration_results(camera_matrix, distortion_coefficients, ret, output_file):
    data = {
        'reprojection_error': ret,
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': distortion_coefficients.tolist(),
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Camera Calibration with Charuco Board')
    parser.add_argument('-i', '--images_folder', type=str, required=True, help='Path to the folder containing calibration images')
    parser.add_argument('-o', '--output_file', type=str, default = "./calibration_results.json", help='Path to save calibration results JSON file')
    # parser.add_argument('--camera_name', type=str, required=True, help='camera name for example D-1-B')
    args = parser.parse_args()

    size = [5, 7]
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    square_length = 0.112
    marker_length = 0.084

    board = create_charuco_board(size, aruco_dict, square_length, marker_length)
    # save_charuco_board_image(board, 'charuco_board.png')

    # ret, camera_matrix, distortion_coefficients = calibrate_camera(args.images_folder, args.camera_name, board, square_length, marker_length)
    ret, camera_matrix, distortion_coefficients = calibrate_camera(args.images_folder, board, square_length, marker_length)

    if ret:
        print("Reprojection Error: " + str(ret))
        save_calibration_results(camera_matrix, distortion_coefficients, ret, args.output_file)
        print("Camera calibration successful. Results saved.")
    else:
        print("Camera calibration failed.")


if __name__ == "__main__":
    main()

