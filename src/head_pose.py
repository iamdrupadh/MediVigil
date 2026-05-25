# ==============================================================================
# MediVigil
# Coder / Lead Developer: Drupad H
# ==============================================================================

import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             
            (0.0, -330.0, -65.0),        
            (-225.0, 170.0, -135.0),     
            (225.0, 170.0, -135.0),      
            (-150.0, -150.0, -125.0),    
            (150.0, -150.0, -125.0)      
        ])

    def estimate_pose(self, image, landmarks2d):
        if len(landmarks2d) != 6:
            return 0.0, 0.0, 0.0

        size = image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype = "double"
        )

        dist_coeffs = np.zeros((4,1))
        image_points = np.array(landmarks2d, dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return yaw, pitch, roll
