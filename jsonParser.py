# import os
# import json
# import pandas as pd

# folder_path_cheating = 'frames/cheatingJson'
# folder_path_not_cheating = 'frames/NotcheatingJson'

# def load_json_data(path):
#     posePoints = []
#     for filename in os.listdir(path):
#         if filename.endswith(".json"):
#             file_path = os.path.join(path, filename)
#             with open(file_path, "r") as f:
#                 json_data = json.load(f)
#                 posePoints.append(json_data['people'])
#     return posePoints

# def extract_pose_keypoints(objects, obj_class):
#     keypoints_list = []
#     for people in objects:
#         for obj in people:
#             keypoints = obj.get("pose_keypoints_2d")
#             if keypoints:
#                 keypoints_list.append({obj_class: keypoints})
#     return keypoints_list

# data_for_cheating = pd.DataFrame(extract_pose_keypoints(load_json_data(folder_path_cheating), "cheating"))
# data_for_notcheating = pd.DataFrame(extract_pose_keypoints(load_json_data(folder_path_not_cheating), "Not cheating"))

# combined_df = pd.concat([data_for_cheating, data_for_notcheating])

# # print(combined_df)
# print(data_for_notcheating)
# print(data_for_cheating)

import os
import json
import pandas as pd

folder_path_cheating = 'frames/cheatingJson'
folder_path_not_cheating = 'frames/NotcheatingJson'

def load_json_data(path):
    posePoints = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            with open(file_path, "r") as f:
                json_data = json.load(f)
                posePoints.append(json_data['people'])
    return posePoints

def extract_pose_keypoints(objects, obj_class):
    keypoints_list = []
    for people in objects:
        for obj in people:
            keypoints = obj.get("pose_keypoints_2d")
            if keypoints:
                keypoints_list.append({obj_class: keypoints})
    return keypoints_list

print(extract_pose_keypoints(load_json_data(folder_path_cheating), "cheating"))

data_for_cheating = pd.DataFrame(extract_pose_keypoints(load_json_data(folder_path_cheating), "cheating"))
data_for_notcheating = pd.DataFrame(extract_pose_keypoints(load_json_data(folder_path_not_cheating), "Not cheating"))

# combined_df = pd.concat([data_for_cheating, data_for_notcheating])

# print(combined_df)
# print(data_for_notcheating)
# print(data_for_cheating)

