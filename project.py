import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
from imageio import imwrite
from tensorflow.keras.applications.vgg16 import preprocess_input
from load_models import load_generator
from inference_from_official_weights import main

folder = input(' Image folder:')

if not os.path.isdir('Image2style_gen'):
    os.mkdir('Image2style_gen')

main()

run_item = {
    'res': 256,
    'ckpt_dir': './official-converted/cuda',
    'use_custom_cuda': True,
    'out_fn': None,
}

res = run_item['res']
ckpt_dir = run_item['ckpt_dir']
use_custom_cuda = run_item['use_custom_cuda']
out_fn = run_item['out_fn']
message = f'{res}x{res} with custom cuda' if use_custom_cuda else f'{res}x{res} without custom cuda'
print(message)

resolutions = [4, 8, 16, 32, 64, 128, 256]
feature_maps = [512, 512, 512, 512, 512, 256, 128]
filter_index = resolutions.index(res)
g_params = {
    'z_dim': 512,
    'w_dim': 512,
    'labels_dim': 0,
    'n_mapping': 8,
    'resolutions': resolutions[:filter_index + 1],
    'featuremaps': feature_maps[:filter_index + 1],
}
generator = load_generator(g_params, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)


def image2latent(image_folder, generator_model):
    """
    Creates latent vector for the images in the image folder, in the latent space of StyleGAN.
    The latent vectors are stored in a pickle file.
    :param image_folder: folder containing images to projected
    :param generator_model: StyleGAN's generator instance
    :return lat_dict: a dictionary of image file name and its embedding.
    """
    mapped_latent = generator_model.g_mapping([tf.random.normal(
        shape=[10000, g_params['z_dim']]),
        tf.random.normal(shape=[10000, g_params['labels_dim']])])
    w_broadcast = tf.tile(mapped_latent[:, tf.newaxis],
                          [1, len(resolutions) * 2, 1])
    avg_latent = tf.expand_dims(tf.math.reduce_mean(w_broadcast, axis=0),
                                axis=0)

    # avg_latent is a latent vector which has been averaged over 10000 mapped latent vectors
    # this serves as a good starting point for projection instead of a random mapped latent vector

    vgg16 = tf.keras.applications.VGG16(include_top=False,
                                        input_shape=(256, 256, 3),
                                        weights='imagenet')
    layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv2', 'block4_conv2']
    layer_outputs = [vgg16.get_layer(layer_name).output for layer_name in layer_names]
    exp_vgg16 = tf.keras.Model(inputs=vgg16.input, outputs=layer_outputs)

    # exp_vgg_16 is a tf.keras.Model instance whose outputs we will use in calculating perceptual distance

    inp = tf.keras.layers.Input(shape=[256, 256, 3])
    x = tf.keras.layers.Lambda(preprocess_input)(inp)
    out = exp_vgg16(x)
    mod_vgg16 = tf.keras.Model(inputs=inp, outputs=out)

    # mod_vgg_16 is a tf.keras.Model which is very similar to the exp_vgg_16 model, the only difference being a
    # preprocess layer introduced after the input layer

    inp1 = tf.keras.layers.Input(shape=[14, 512])
    gen_out = generator_model.synthesis(inp1)
    gen_out = tf.keras.layers.Lambda(lambda t: (tf.clip_by_value(t, -1.0, 1.0) + 1.0) * 127.5)(gen_out)
    gen_out = tf.keras.layers.Lambda(lambda t: tf.transpose(t, [0, 2, 3, 1]))(gen_out)
    gen_out_final = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 255.0))(gen_out)
    x = mod_vgg16(gen_out_final)
    outputs = [gen_out_final, x]
    model = tf.keras.Model(inputs=inp1, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01,
                                   beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # model is again a tf.keras.Model instance which has concatenated the generator with mod_vgg_16 model
    # this module will be used for projection

    lat_dict = {}

    # lat_dict is a dictionary which contains latent codes for each image

    iterations = 6000

    # the variable 'iterations' defines the number of iterations the projection will take place for every image

    bar = tqdm(os.listdir(image_folder))
    for file_name in bar:
        if 'jpg' in file_name or 'png' in file_name:
            image = cv2.imread(image_folder + '/' + file_name)
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            imwrite('Image2style_gen/' + 'real_' + file_name, image)

            # real image i.e. the image to be projected is saved in the Image2style_gen folder.

            image = image.astype(np.float32)
            image = np.expand_dims(image, 0)
            og_vgg_features = mod_vgg16.predict(image)

            # og_vgg_features contains vgg16 features corresponding to the original image.

            latent_variable = tf.Variable(avg_latent)

            # latent_variable is a tf.Variable instance which will be updated at every projection step

            for j in range(iterations):
                bar.set_description(desc=file_name + " %i/6000" % j)
                bar.refresh()
                with tf.GradientTape() as tape:
                    tape.watch(latent_variable)
                    layer_outputs = model(latent_variable)
                    loss1 = 0
                    vgg_features = layer_outputs[1]
                    for feature_index in range(len(vgg_features)):
                        array_shape = og_vgg_features[feature_index].shape
                        n = 1
                        for i in array_shape:
                            n = n * i
                        loss1 += (1 / n) * tf.norm(
                            tf.subtract(og_vgg_features[feature_index], vgg_features[feature_index]),
                            ord='euclidean')
                    n = 1
                    array_shape = image.shape
                    for i in array_shape:
                        n = n * i
                    gen_im = layer_outputs[0]
                    loss2 = (1 / n) * tf.norm(tf.subtract(image, gen_im), ord='euclidean')
                    total_loss = loss1 + loss2
                grads = tape.gradient(total_loss, [latent_variable])
                opt.apply_gradients(zip(grads, [latent_variable]))

                # here loss has been implemented according to the paper

                if (j + 1) % 6000 == 0:
                    gen_im = generator_model.synthesis(latent_variable)
                    gen_im = tf.transpose(gen_im, [0, 2, 3, 1])
                    gen_im = (tf.clip_by_value(gen_im, -1.0, 1.0) + 1.0) * 127.5
                    gen_im = tf.cast(gen_im, tf.uint8)
                    imwrite('Image2style_gen/' + 'step_' + str(j + 1) + '_' + file_name,
                            gen_im[0].numpy())

                    # at the end of all the projection steps the projected image is stored in the Image2style_gen.

            latent = latent_variable
            lat_dict[file_name] = latent

            # lat_dict stores the latent corresponding to each filename.

            pickle.dump(lat_dict, open(image_folder + '/' + 'latent_codes_' + image_folder + '.pkl', 'wb'))

    return lat_dict


latent_dict = image2latent(folder, generator)
