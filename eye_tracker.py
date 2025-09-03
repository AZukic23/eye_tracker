import cv2
import dlib
import numpy as np
import math

class EyeTracker:
    def __init__(self):
        # Initialize face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Eye tracking parameters
        self.eye_center = None
        self.gaze_point = (320, 240)  # Default center point
        
    def get_eye_landmarks(self, landmarks):
        """Extract eye landmarks from facial landmarks"""
        left_eye = []
        right_eye = []
        
        # Left eye landmarks (points 36-41)
        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
            
        # Right eye landmarks (points 42-47)
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))
            
        return left_eye, right_eye
    
    def get_eye_center(self, eye_landmarks):
        """Calculate the center of an eye"""
        x_coords = [point[0] for point in eye_landmarks]
        y_coords = [point[1] for point in eye_landmarks]
        
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        
        return (center_x, center_y)
    
    def detect_pupil(self, eye_region, eye_landmarks):
        """Detect pupil in eye region"""
        if eye_region is None:
            return None
            
        # Convert to grayscale
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        
        # Find the darkest point (pupil)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        return min_loc
    
    def extract_eye_region(self, frame, eye_landmarks):
        """Extract eye region from frame"""
        # Get bounding rectangle for eye
        x_coords = [point[0] for point in eye_landmarks]
        y_coords = [point[1] for point in eye_landmarks]
        
        x = min(x_coords) - 10
        y = min(y_coords) - 10
        w = max(x_coords) - min(x_coords) + 20
        h = max(y_coords) - min(y_coords) + 20
        
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        return frame[y:y+h, x:x+w], (x, y)
    
    def calculate_gaze_point(self, left_pupil, right_pupil, left_offset, right_offset):
        """Calculate gaze point based on pupil positions"""
        if left_pupil is None or right_pupil is None:
            return self.gaze_point
            
        # Adjust pupil coordinates to frame coordinates
        left_pupil_adjusted = (left_pupil[0] + left_offset[0], left_pupil[1] + left_offset[1])
        right_pupil_adjusted = (right_pupil[0] + right_offset[0], right_pupil[1] + right_offset[1])
        
        # Calculate average pupil position
        avg_x = (left_pupil_adjusted[0] + right_pupil_adjusted[0]) // 2
        avg_y = (left_pupil_adjusted[1] + right_pupil_adjusted[1]) // 2
        
        # Map to screen coordinates (simplified mapping)
        screen_x = int(avg_x * 1.5)  # Scale factor for X
        screen_y = int(avg_y * 1.2)  # Scale factor for Y
        
        # Keep within frame bounds
        screen_x = max(0, min(640, screen_x))
        screen_y = max(0, min(480, screen_y))
        
        return (screen_x, screen_y)
    
    def run(self):
        """Main tracking loop"""
        print("Eye Tracker Started! Press 'q' to quit.")
        print("Make sure you have downloaded the shape_predictor_68_face_landmarks.dat file!")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            for face in faces:
                # Get facial landmarks
                landmarks = self.predictor(gray, face)
                
                # Get eye landmarks
                left_eye, right_eye = self.get_eye_landmarks(landmarks)
                
                # Extract eye regions
                left_eye_region, left_offset = self.extract_eye_region(frame, left_eye)
                right_eye_region, right_offset = self.extract_eye_region(frame, right_eye)
                
                # Detect pupils
                left_pupil = self.detect_pupil(left_eye_region, left_eye)
                right_pupil = self.detect_pupil(right_eye_region, right_eye)
                
                # Calculate gaze point
                self.gaze_point = self.calculate_gaze_point(left_pupil, right_pupil, left_offset, right_offset)
                
                # Draw eye landmarks
                for point in left_eye + right_eye:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)
                
                # Draw pupil positions if detected
                if left_pupil:
                    pupil_pos = (left_pupil[0] + left_offset[0], left_pupil[1] + left_offset[1])
                    cv2.circle(frame, pupil_pos, 3, (255, 0, 0), -1)
                    
                if right_pupil:
                    pupil_pos = (right_pupil[0] + right_offset[0], right_pupil[1] + right_offset[1])
                    cv2.circle(frame, pupil_pos, 3, (255, 0, 0), -1)
            
            # Draw gaze indicator (large circle)
            cv2.circle(frame, self.gaze_point, 20, (0, 0, 255), 3)
            cv2.circle(frame, self.gaze_point, 5, (0, 0, 255), -1)
            
            # Add instructions
            cv2.putText(frame, "Red circle follows your gaze", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Eye Tracker', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run()