"""
Camera calibration utility for distance estimation
"""

import cv2
import numpy as np
import glob
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CameraCalibrator:
    """
    Camera calibration for accurate distance measurement
    """
    
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.focal_length = None
        
    def calibrate_chessboard(self, images_path: str, 
                            pattern_size: tuple = (9, 6),
                            square_size: float = 0.025):
        """
        Calibrate camera using chessboard pattern
        
        Args:
            images_path: Path to calibration images
            pattern_size: Number of inner corners (width, height)
            square_size: Size of square in meters
        """
        logger.info("Starting camera calibration...")
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []
        imgpoints = []
        
        # Get list of calibration images
        images = glob.glob(str(Path(images_path) / '*.jpg'))
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                logger.info(f"Found corners in {fname}")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        
        # Calculate focal length in pixels
        self.focal_length = (mtx[0, 0] + mtx[1, 1]) / 2
        
        logger.info(f"Calibration complete. Focal length: {self.focal_length:.2f} pixels")
        logger.info(f"Camera matrix:\n{mtx}")
        
        return mtx, dist
    
    def calibrate_focal_length_manual(self, known_distance: float, 
                                     known_height: float,
                                     pixel_height: float) -> float:
        """
        Manually calibrate focal length
        
        Args:
            known_distance: Actual distance to object (meters)
            known_height: Actual height of object (meters)
            pixel_height: Measured pixel height
            
        Returns:
            Calculated focal length
        """
        self.focal_length = (pixel_height * known_distance) / known_height
        logger.info(f"Manual calibration: Focal length = {self.focal_length:.2f}")
        return self.focal_length
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort image using calibration parameters
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.warning("Camera not calibrated. Returning original image.")
            return image
        
        h, w = image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, new_camera_mtx
        )
        
        return undistorted
    
    def save_calibration(self, filepath: str):
        """Save calibration data to file"""
        data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'focal_length': self.focal_length
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration data from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        self.focal_length = data['focal_length']
        
        logger.info(f"Calibration loaded from {filepath}")
        logger.info(f"Focal length: {self.focal_length:.2f}")

def main():
    """Example usage"""
    calibrator = CameraCalibrator()
    
    # Option 1: Chessboard calibration
    # calibrator.calibrate_chessboard('calibration_images/')
    
    # Option 2: Manual calibration
    # Place an object of known height (e.g., 1m) at known distance (e.g., 5m)
    # Measure its pixel height from the image
    focal_length = calibrator.calibrate_focal_length_manual(
        known_distance=5.0,  # meters
        known_height=1.0,    # meters
        pixel_height=125     # pixels (example)
    )
    
    print(f"Focal Length: {focal_length:.2f} pixels")
    
    # Save calibration
    calibrator.save_calibration('camera_calibration.pkl')

if __name__ == "__main__":
    main()