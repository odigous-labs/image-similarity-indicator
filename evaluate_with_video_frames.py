'''
Within this script, focus to pass set of frames to the pre-trained siamese model.
----
If need to run this script seperately, then can edit the relevant input file path and output file path.

If need to use this script within another code then can import the script and call the functions with relevant arguments.

We define the positive and negative as follows for this project:
    pos + pos -> 1
    pos + neg -> 0
'''

import frames_to_siamese
from keras.models import model_from_json
from pathlib import Path
import os
from shutil import copyfile


def write_images(prediction_array, video_frame_path_array, generated_video_frame_path, summary_output_frame_path):

    for index in range(len(video_frame_path_array)):
        if (prediction_array[index][0]>0.4):
            frame_full_path = os.path.join(generated_video_frame_path, video_frame_path_array[index])
            frame_full_dest = os.path.join(summary_output_frame_path, video_frame_path_array[index])
            copyfile(frame_full_path, frame_full_dest)


def predict_similarity_for_ooi(model, generated_video_frame_path,object_of_interest_path, summary_output_frame_path, image_size ):

    frame_input_array, ooi_input_array, video_frames_path_array = frames_to_siamese.import_frames_to_siamese(
        generated_video_frame_path,
        object_of_interest_path,
        image_size
    )

    prediction_array = model.predict([frame_input_array ,ooi_input_array], verbose=1)
    print("Predictions")
    print(prediction_array)
    write_images(prediction_array, video_frames_path_array, generated_video_frame_path, summary_output_frame_path)
    print("Successfully Written Selectred Images !!!")

if __name__ == "__main__":

    image_size = 224
    data = os.path.abspath("data")
    video_data = os.path.join(data, "video_data")

    model_directory_path = os.path.join(data, "Models")
    model_id = "trained_6"
    model_name = model_id + ".json"
    weights_name = model_id + ".h5"
    model_path = os.path.join(model_directory_path, model_name)
    weights_path = os.path.join(model_directory_path, weights_name)
    f = Path(model_path)
    feature_model_structure = f.read_text()
    feature_model = model_from_json(feature_model_structure)
    feature_model.load_weights(weights_path)
    feature_model.summary()

    generated_video_frame_path = os.path.join(video_data, "generated_frames")
    object_of_interest_path = os.path.join(video_data, "object_of_interest")
    summary_output_frame_path = os.path.join(video_data, "output_frame_path")

    predict_similarity_for_ooi(
        feature_model,
        generated_video_frame_path,
        object_of_interest_path,
        summary_output_frame_path,
        image_size
    )
