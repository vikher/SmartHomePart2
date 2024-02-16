# -*- coding: utf-8 -*-

import cv2
import os
import tensorflow as tf
import frameextractor as fe
import handshape_feature_extractor as hfe
import csv
import re as regex

# Constants
TRAIN_DATA_PATH = "traindata/"
VIDEO_LOCATIONS = ["test/"]
RESULTS_FILE_PATH = 'Results.csv'
MAX_MUTATIONS = 5
MAX_TEST_COUNT = 51

class GestureDetail:
    def __init__(self, gesture_key, gesture_name, output_label):
        self.gesture_key = gesture_key
        self.gesture_name = gesture_name
        self.output_label = output_label

class GestureFeature:
    def __init__(self, gesture_detail: GestureDetail, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature

def extract_feature(location, input_file, mid_frame_counter):
    middle_image = cv2.imread(fe.frameExtractor(location + input_file, location + "frames/", mid_frame_counter),
                              cv2.IMREAD_GRAYSCALE)
    response = hfe.HandShapeFeatureExtractor.extract_feature(hfe.HandShapeFeatureExtractor.get_instance(),
                                                             middle_image)
    return response

def decide_gesture_by_file_name(gesture_file_name):
    for x in gesture_details:
        if x.gesture_key == gesture_file_name.split('_')[0]:
            return x
    return None

def decide_gesture_by_name(lookup_gesture_name):
    for x in gesture_details:
        if x.gesture_name.replace(" ", "").lower() == lookup_gesture_name.lower():
            return x
    return None

def validate_mutate_recognition(gesture_file_name, extracted_feature_vector, calc_gesture_detail: GestureDetail):
    actual_gesture = regex.search('-H-(.*?).mp4', gesture_file_name)

    if actual_gesture is None:
        actual_gesture = gesture_file_name.split('_')[0]
        add_to_vector = False
    else:
        actual_gesture = actual_gesture.group(1)
        add_to_vector = True

    if calc_gesture_detail.gesture_name == actual_gesture or calc_gesture_detail.gesture_key == actual_gesture:
        if add_to_vector:
            featureVectorList.append(GestureFeature(calc_gesture_detail, extracted_feature_vector))
    else:
        print("mutating vector set for gesture: " + actual_gesture + " for gesture file: " + gesture_file_name)
        actual_gesture_detail = decide_gesture_by_name(actual_gesture)
        if actual_gesture_detail is not None:
            featureVectorList.append(GestureFeature(actual_gesture_detail, extracted_feature_vector))
        else:
            print(
                "Gesture detail not decoded for gesture: " + actual_gesture + " for gesture file: " + gesture_file_name)
        return True
    return False

def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)

    re_run = True
    max_mutations = 0
    gesture_detail: GestureDetail = GestureDetail("", "", "")
    while re_run and max_mutations < MAX_MUTATIONS:
        cos_sin = 1
        position = 0
        cursor = 0
        for idx, featureVector in enumerate(featureVectorList):
            calc_cos_sin = tf.keras.losses.cosine_similarity(
                video_feature,
                featureVector.extracted_feature,
                axis=-1
            )
            if calc_cos_sin < cos_sin:
                cos_sin = calc_cos_sin
                position = idx
            cursor += 1

        gesture_detail = featureVectorList[position].gesture_detail
        print(gesture_file_name + " calculated gesture " + gesture_detail.gesture_name)
        # re_run = validate_mutate_recognition(gesture_file_name, video_feature, gesture_detail)
        re_run = False
        if re_run:
            max_mutations += 1
    return gesture_detail

def read_train_data():
    feature_vector_list = []
    count = 0
    for file in os.listdir(TRAIN_DATA_PATH):
        if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
            feature_vector_list.append(GestureFeature(decide_gesture_by_file_name(file),
                                                      extract_feature(TRAIN_DATA_PATH, file, count)))
            count += 1
    return feature_vector_list

def write_results_to_csv(results):
    with open(RESULTS_FILE_PATH, 'w', newline='') as results_file:
        train_data_writer = csv.writer(results_file)
        train_data_writer.writerows(results)

def main():
    global gesture_details
    global featureVectorList

    gesture_details = [GestureDetail("Num0", "0", "0"), GestureDetail("Num1", "1", "1"),
                       GestureDetail("Num2", "2", "2"), GestureDetail("Num3", "3", "3"),
                       GestureDetail("Num4", "4", "4"), GestureDetail("Num5", "5", "5"),
                       GestureDetail("Num6", "6", "6"), GestureDetail("Num7", "7", "7"),
                       GestureDetail("Num8", "8", "8"), GestureDetail("Num9", "9", "9"),
                       GestureDetail("FanDown", "Decrease Fan Speed", "10"),
                       GestureDetail("FanOn", "FanOn", "11"), GestureDetail("FanOff", "FanOff", "12"),
                       GestureDetail("FanUp", "Increase Fan Speed", "13"),
                       GestureDetail("LightOff", "LightOff", "14"), GestureDetail("LightOn", "LightOn", "15"),
                       GestureDetail("SetThermo", "SetThermo", "16")
                       ]

    featureVectorList = read_train_data()

    results = []
    test_count = 0

    for video_location in VIDEO_LOCATIONS:
        for test_file in os.listdir(video_location):
            if not test_file.startswith('.') and not test_file.startswith('frames') \
                    and not test_file.startswith('results'):
                recognized_gesture_detail = determine_gesture(video_location, test_file, test_count)
                test_count += 1

                results.append([recognized_gesture_detail.output_label])

                if test_count >= MAX_TEST_COUNT:
                    break

        if test_count >= MAX_TEST_COUNT:
            break

    write_results_to_csv(results)

if __name__ == "__main__":
    main()
