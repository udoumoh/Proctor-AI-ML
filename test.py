import os
import sys
import cv2
import os
import sys
import numpy as np
import argparse

dir_path = 'C:/Users/JOHN/Documents/openpose/build'
sys.path.append(dir_path + '/python/openpose/Release')
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' +  dir_path + '/bin;'

import pyopenpose as op

# Set OpenPose parameters
params = dict()
params["model_folder"] = "C:/Users/JOHN/Documents/openpose/models"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
# params["tracking"] = 5
params["number_people_max"] = 3

# Create OpenPose instance
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Create video capture object
cap = cv2.VideoCapture('C:/Users/JOHN/Videos/project/test.MOV')

# Process each frame of the video
while True:
    # Read a frame
    ret, image = cap.read()
    if not ret:
        break
    
    # Process the frame with OpenPose
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    
    # Get pose keypoints and output image
    keypoints = datum.poseKeypoints
    output_image = datum.cvOutputData
    
    # Display the output image
    cv2.imshow('OpenPose Output', output_image)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()