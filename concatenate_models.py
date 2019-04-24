'''
Within this script, call the fit function of the model (siamese for our case) and save the model and weights.
----
If need to run this script seperately, then can edit the relevant input and output file path(s).

If need to use this script within another code then can import the script and call the functions with relevant arguments.

We define the positive and negative as follows for this project:
    pos + pos -> 1
    pos + neg -> 0
'''
from keras.applications import VGG16
import os
import data_handler as dh
import siamese_model_structure
from keras.models import Model
import vgg_model

def fit_model(epochs, batch_size, model_save_directory, model_id, input_model_1, input_model_2, labeled_output):

    vec_1, vec_2, vgg_1, vgg_2 = vgg_model.return_vgg_vectors()
    pred = siamese_model_structure.siamese_model(vec_1, vec_2)
    model = Model(inputs=[vgg_1.input, vgg_2.input], outputs=pred)
    model.summary()

    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
    # from the lc
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])

    model.fit(
        [input_model_1, input_model_2],
        labeled_output,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    # for save the model
    model_json = model.to_json()

    model_name = model_id  + ".json"
    weights_name = model_id  + ".h5"
    model_path = os.path.join(model_save_directory, model_name)
    weights_path = os.path.join(model_save_directory, weights_name)

    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path)

    print('Successfully saved the model files in path: '+model_save_directory)

if __name__ == "__main__":

    data = os.path.abspath("data")
    test_image_directory = os.path.join(data, "test", "jpg")
    model_directory_path = os.path.join(data, "Models")
    image_directory = os.path.join(data, "images", "jpg")
    image_size = 224
    epochs = 50
    mini_batch_size = 32

    print('Creating triples...')
    triples = dh.create_image_triples(image_directory)

    print('Loading images...')
    input_model_1, input_model_2, labeled_output = dh.load_image_triplets(image_dir=image_directory,
                                         image_triples=triples,
                                         image_size=image_size, shuffle=True)
    print('y', labeled_output.shape)
    print('lhs', input_model_1.shape)
    print('rhs', input_model_2.shape)

    if not os.path.isdir(model_directory_path):
        os.makedirs(model_directory_path)
        print('Model directory created!')

    model_id = "trained_6"

    fit_model(epochs, mini_batch_size, model_directory_path, model_id, input_model_1, input_model_2, labeled_output)