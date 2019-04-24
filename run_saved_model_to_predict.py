'''
Within this script, focus to run saved siamese model to do predictions and validate.
----
If need to run this script seperately, then can edit the relevant input file path and output file path.

If need to use this script within another code then can import the script and call the functions with relevant arguments.

We define the positive and negative as follows for this project:
    pos + pos -> 1
    pos + neg -> 0
'''

from keras.models import model_from_json
from pathlib import Path
import data_handler as dh
import os


def test_on_test_data(model_path, weights_path, test_image_directory, image_size):
    # Load the json file that contains the model's structure
    f = Path(model_path)
    feature_model_structure = f.read_text()
    # Recreate the Keras model object from the json data
    feature_model = model_from_json(feature_model_structure)
    # Re-load the model's trained weights
    feature_model.load_weights(weights_path)

    feature_model.summary()

    triples = dh.create_image_triples(test_image_directory)

    lhs, rhs, y = dh.load_image_triplets(image_dir=test_image_directory,
                                         image_triples=triples,
                                         image_size=image_size, shuffle=True)

    pred = feature_model.predict([lhs, rhs])

    print("Predictions:")
    print(pred)
    print("Real y values:")
    print(y)

if __name__ == "__main__":
    image_size = 224
    data = os.path.abspath("data")
    test_image_directory = os.path.join(data, "test", "jpg")
    model_directory_path = os.path.join(data, "Models")
    model_id = "trained_6"

    model_name = model_id + ".json"
    weights_name = model_id + ".h5"

    model_path = os.path.join(model_directory_path, model_name)
    weights_path = os.path.join(model_directory_path, weights_name)

    test_on_test_data(model_path, weights_path, test_image_directory, image_size)