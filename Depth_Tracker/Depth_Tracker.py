import cv2
import time
import math as m
import mediapipe as mp


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


if __name__ == "__main__":
    # For webcam input replace file name with 0.
    file_name = 'input.mp4'
    cap = cv2.VideoCapture(0 )

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern. 
             
        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # Right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        # Left hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        # Right hip.
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
        # Right knee.
        r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
        r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
        # Left knee.
        l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
        # Right ankle.
        r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)
        # Left ankle.
        l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
        # Right heel.
        r_heel_x = int(lm.landmark[lmPose.RIGHT_HEEL].x * w)
        r_heel_y = int(lm.landmark[lmPose.RIGHT_HEEL].y * h)
        # Left heel.
        l_heel_x = int(lm.landmark[lmPose.LEFT_HEEL].x * w)
        l_heel_y = int(lm.landmark[lmPose.LEFT_HEEL].y * h)
        # Right foot.
        r_foot_x = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x * w)
        r_foot_y = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].y * h)
        # Left foot.
        l_foot_x = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w)
        l_foot_y = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].y * h)

        # Calculate distance between left shoulder and right shoulder points.
        #offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        #if offset < 100:
         #   cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        #else:
        #    cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        # Calculate angles.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        thigh_inclination = findAngle(l_hip_x, l_hip_y, l_knee_x, l_knee_y)

        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, yellow, -1)
        
        cv2.circle(image, (r_hip_x, r_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        
        cv2.circle(image, (r_knee_x, r_knee_y), 7, yellow, -1)
        cv2.circle(image, (l_knee_x, l_knee_y), 7, yellow, -1)
        
        cv2.circle(image, (r_ankle_x, r_ankle_y), 7, yellow, -1)
        cv2.circle(image, (l_ankle_x, l_ankle_y), 7, yellow, -1)
        
        cv2.circle(image, (r_heel_x, r_heel_y), 7, yellow, -1)
        cv2.circle(image, (l_heel_x, l_heel_y), 7, yellow, -1)
        
        cv2.circle(image, (r_foot_x, r_foot_y), 7, yellow, -1)
        cv2.circle(image, (l_foot_x, l_foot_y), 7, yellow, -1)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        #cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        #cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        #neck_inclination < 40 and torso_inclination < 10
        if True :
            bad_frames = 0
            good_frames += 1
            
            #cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(thigh_inclination)), (l_knee_x + 10, l_knee_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

            # Join landmarks.
            #cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            #cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            #cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            #cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
            

            #wireframe
         
            #hips
            cv2.line(image, (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), green, 4)
            #knee to hip
            cv2.line(image, (l_knee_x, l_knee_y), (l_hip_x, l_hip_y), green, 4)
            cv2.line(image, (r_knee_x, r_knee_y), (r_hip_x, r_hip_y), green, 4)
            #ankle to knee
            cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), green, 4)
            cv2.line(image, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), green, 4)
            #HEEL to ankle
            cv2.line(image, (l_heel_x, l_heel_y), (l_ankle_x, l_ankle_y), green, 4)
            cv2.line(image, (r_heel_x, r_heel_y), (r_ankle_x, r_ankle_y), green, 4)
            #HEEL to TOE
            cv2.line(image, (l_heel_x, l_heel_y), (l_foot_x, l_foot_y), green, 4)
            cv2.line(image, (r_heel_x, r_heel_y), (r_foot_x, r_foot_y), green, 4)
            #HEEL to TOE
            cv2.line(image, (l_ankle_x, l_ankle_y), (l_foot_x, l_foot_y), green, 4)
            cv2.line(image, (r_ankle_x, r_ankle_y), (r_foot_x, r_foot_y), green, 4)
            

        else:
            good_frames = 0
            bad_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

            # Join landmarks.
            #cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            #cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            #cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            #cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)
            

            #new
            #cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        # Pose time.
       # if good_time > 0:
        #    time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
         #   cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        #else:
         ##  cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If you stay in bad posture for more than 3 minutes (180s) send an alert.
       # if bad_time > 180:
        #    sendWarning()
        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()