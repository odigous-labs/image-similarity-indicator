'''
Within this script, focus to return two feature vector layers of the vgg model
----
If need to use this script within another code then can import the script and call the function(s).

We define the positive and negative as follows for this project:
    pos + pos -> 1
    pos + neg -> 0
'''



from keras.applications import VGG16

def return_vgg_vectors():
    vgg_model_1 = VGG16(weights='imagenet', include_top=True)
    vgg_model_2 = VGG16(weights='imagenet', include_top=True)

    for layer in vgg_model_1.layers:
        layer.trainable = False
        layer.name = layer.name + "_1"
    for layer in vgg_model_2.layers:
        layer.trainable = False
        layer.name = layer.name + "_2"

    vgg_model_1.summary()

    v1 = vgg_model_1.get_layer("flatten_1").output
    v2 = vgg_model_2.get_layer("flatten_2").output

    return v1, v2, vgg_model_1, vgg_model_2






