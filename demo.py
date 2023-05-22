import os
import sys
import cv2
import numpy as np
import argparse
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree


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
params["number_people_max"] = 10

# Create OpenPose instance
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()

# Starting XGBoost
model = XGBClassifier()
xgboost_model_path = "C:/Users/JOHN/Desktop/gembacud/Cheating-Detection/CheatDetection/XGB_BiCD_Tuned_GPU_05.model"
model.load_model(xgboost_model_path)
model.set_params(**{"predictor": "gpu_predictor"})

image_path = "C:/Users/JOHN/Desktop/examMalpracticeDetection/test.jpg"
image = cv2.imread(image_path)
# Process the frame with OpenPose
datum.cvInputData = image
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    
# Get pose keypoints and output image
keypoints = datum.poseKeypoints
output_image = datum.cvOutputData


def GetColumnNames(dim=None):
    # columnNames = []
    # for i in range(25):
    #     columnNames.append("kp" + str(i) + "_X")
    #     columnNames.append("kp" + str(i) + "_Y")
    #     columnNames.append("kp" + str(i) + "_Z")
    # return columnNames
    if dim == "x" or dim == "X":
        return
        ['kp0_X', 'kp1_X', 'kp2_X', 'kp3_X', 'kp4_X', 'kp5_X', 'kp6_X', 'kp7_X', 'kp8_X', 'kp9_X', 'kp10_X', 'kp11_X', 'kp12_X',
            'kp13_X', 'kp14_X', 'kp15_X', 'kp16_X', 'kp17_X', 'kp18_X', 'kp19_X', 'kp20_X', 'kp21_X', 'kp22_X', 'kp23_X', 'kp24_X']

    if dim == "y" or dim == "Y":
        return ['kp0_Y', 'kp1_Y', 'kp2_Y', 'kp3_Y', 'kp4_Y', 'kp5_Y', 'kp6_Y', 'kp7_Y', 'kp8_Y', 'kp9_Y', 'kp10_Y', 'kp11_Y',
                'kp12_Y', 'kp13_Y', 'kp14_Y', 'kp15_Y', 'kp16_Y', 'kp17_Y', 'kp18_Y', 'kp19_Y', 'kp20_Y', 'kp21_Y', 'kp22_Y', 'kp23_Y', 'kp24_Y']

    if dim == "z" or dim == "Z":
        return ['kp0_Z', 'kp1_Z', 'kp2_Z', 'kp3_Z', 'kp4_Z', 'kp5_Z', 'kp6_Z', 'kp7_Z', 'kp8_Z', 'kp9_Z', 'kp10_Z', 'kp11_Z', 'kp12_Z',
                'kp13_Z', 'kp14_Z', 'kp15_Z', 'kp16_Z', 'kp17_Z', 'kp18_Z', 'kp19_Z', 'kp20_Z', 'kp21_Z', 'kp22_Z', 'kp23_Z', 'kp24_Z']

    return ['kp0_X', 'kp0_Y', 'kp0_Z', 'kp1_X', 'kp1_Y', 'kp1_Z',
            'kp2_X', 'kp2_Y', 'kp2_Z', 'kp3_X', 'kp3_Y', 'kp3_Z', 'kp4_X',
            'kp4_Y', 'kp4_Z', 'kp5_X', 'kp5_Y', 'kp5_Z', 'kp6_X', 'kp6_Y',
            'kp6_Z', 'kp7_X', 'kp7_Y', 'kp7_Z', 'kp8_X', 'kp8_Y', 'kp8_Z',
            'kp9_X', 'kp9_Y', 'kp9_Z', 'kp10_X', 'kp10_Y', 'kp10_Z',
            'kp11_X', 'kp11_Y', 'kp11_Z', 'kp12_X', 'kp12_Y', 'kp12_Z',
            'kp13_X', 'kp13_Y', 'kp13_Z', 'kp14_X', 'kp14_Y', 'kp14_Z',
            'kp15_X', 'kp15_Y', 'kp15_Z', 'kp16_X', 'kp16_Y', 'kp16_Z',
            'kp17_X', 'kp17_Y', 'kp17_Z', 'kp18_X', 'kp18_Y', 'kp18_Z',
            'kp19_X', 'kp19_Y', 'kp19_Z', 'kp20_X', 'kp20_Y', 'kp20_Z',
            'kp21_X', 'kp21_Y', 'kp21_Z', 'kp22_X', 'kp22_Y', 'kp22_Z',
            'kp23_X', 'kp23_Y', 'kp23_Z', 'kp24_X', 'kp24_Y', 'kp24_Z']



def NormalizePose(pose, flipY=True):
    convertedPose = []
    # Compute the Maximum bounds in dimensions X and Y of the pose
    maxX, minX = -math.inf, math.inf
    maxY, minY = -math.inf, math.inf
    for keyPoint in pose:
        if keyPoint[2] == 0:
            continue
        if keyPoint[0] > maxX:
            maxX = keyPoint[0]
        if keyPoint[0] < minX:
            minX = keyPoint[0]
        if keyPoint[1] > maxY:
            maxY = keyPoint[1]
        if keyPoint[1] < minY:
            minY = keyPoint[1]
    frameDiffX = maxX - minX
    frameDiffY = maxY - minY
    # Convert the Coordinates to normalized values
    # NOTE: The Origin of the KeyPoints is located at the topleft of the image and is forcibly flipped around the X axis to
    # reflect the rectangular coordinate system where its logical Origin is now at the bottomleft.
    for keyPoint in pose:
        convertedKeyPoint = [0, 0, 0]
        if keyPoint[2] == 0:
            convertedPose.append(convertedKeyPoint)
            continue
        convertedKeyPoint[0] = (keyPoint[0] - minX) / (frameDiffX)
        if flipY == True:
            convertedKeyPoint[1] = (keyPoint[1] - minY) / (frameDiffY)
        else:
            convertedKeyPoint[1] = (maxY - keyPoint[1]) / (frameDiffY)
        convertedKeyPoint[2] = keyPoint[2]
        convertedPose.append(convertedKeyPoint)
    return convertedPose


def ReshapePoseCollection(poseCollection):
    numPoses, numKeyPoints, KeyPointVector = (
        poseCollection.shape[0],
        poseCollection.shape[1],
        poseCollection.shape[2],
    )
    poseCollection = np.reshape(
        poseCollection, (numPoses, numKeyPoints * KeyPointVector)
    )
    return poseCollection


def ConvertToDataFrame(poseCollection, label=None):
    columnNames = GetColumnNames()
    poseDF = pd.DataFrame(poseCollection, columns=columnNames)
    if label is not None:
        poseDF["label"] = label
    return poseDF


def DetectCheat(ShowPose=True, img=None):
        poseCollection = keypoints
        detectedPoses = []
        cheating = False
        if ShowPose == True:
            OutputImage = datum.cvOutputData
        else:
            OutputImage = image
        if poseCollection.ndim != 0:
            original_posecollection = copy.deepcopy(poseCollection)
            poseCollection = NormalizePoseCollection(poseCollection)
            poseCollection = ReshapePoseCollection(poseCollection)
            poseCollection = ConvertToDataFrame(poseCollection)
            preds = model.predict(poseCollection)
            for idx, pred in enumerate(preds):
                if pred:
                    cheating = True
                    OutputImage = DrawBoundingRectangle(
                        OutputImage, GetBoundingBoxCoords(
                            original_posecollection[idx])
                    )

            # for pose in poseCollection:
            #     # * Normalize Pose Collection
            #     pose = NormalizePose(pose)
            #     #  * Reshaping of Pose Collection
            #     pose = ReshapePose(pose)
            #     # * Creating a Pose Collection DataFrame
            #     pose = ConvertToDataFrame(pose)
            #     # * Classify Pose
            #     pred = self.model.predict(pose)
                # * Draw BoundingBox
                # if pred:
                #     OutputImage = DrawBoundingRectangle(
                #         OutputImage, GetBoundingBoxCoords(original_pose)
                #     )
                #     cheating = True

        return OutputImage, cheating