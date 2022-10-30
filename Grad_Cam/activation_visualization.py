# %%

# from librosa.display import cmap
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

# Display
import librosa
import matplotlib.pyplot as plt
from skimage.transform import resize
# %%


def make_gradcam_heatmap(img_array, model, last_conv_layer_name,
                         pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input
    # image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 &
    # 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# %% Load Dataset:


json_file_codec = os.path.join('_data/tf_dataset_Database/DLNet_config.json')
with open(json_file_codec, "r") as read_file:
    config = json.load(read_file)

path_dataset = "_data/tf_dataset_Database/Database_test_set"
ds = tf.data.experimental.load(path_dataset,
                               (tf.TensorSpec(config['input_shape'],
                                              dtype=tf.float32, name=None),
                                tf.TensorSpec(len(config['classes']),
                                              dtype=tf.uint8, name=None)),
                               compression='GZIP')

ds_list = list(ds.as_numpy_iterator())

# %% Read model:
best_model = keras.models.load_model('best_model.h5')
best_model.summary()
last_conv_layer_name = "conv_layer4"
# Remove last layer's softmax
best_model.layers[-1].activation = None

# %% Calculate heatmap

idx = 600
sample = ds_list[idx]
spec = sample[0]
label_arr = sample[1]
ext_spec = spec[np.newaxis, :]
padded_spec = np.pad(ds_list[0][0][:, :, 0], ((93, 0), (0, 0)))
librosa.display.specshow(padded_spec,
                         sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='linear')


preds = best_model.predict(ext_spec)
pred_idx = int(np.argmax(preds, axis=1))
print(f"Predicted: {config['classes'][pred_idx]}")
print(f"Actual: {config['classes'][int(np.argmax(label_arr))]}")

heatmap = make_gradcam_heatmap(ext_spec, best_model, last_conv_layer_name,
                               pred_index=pred_idx)
resized_heatmap = resize(heatmap, (spec.shape[0], spec.shape[1]),
                         anti_aliasing=True)


# %%

fig = plt.figure(figsize=(12, 12))
plt.title(f"{config['classes'][int(np.argmax(ds_list[idx][1]))]}",
          fontsize=16)
plt.imshow(np.flipud(spec), cmap='gray', interpolation='none')
plt.imshow(np.flipud(resized_heatmap), alpha=0.5, cmap='hot', interpolation='none')
plt.show()

# %%


heatmap_lst = []
spectrogram_lst = []
label_lst = []
for idx in range(0, len(ds_list), 100):
    sample = ds_list[idx]
    spec = sample[0]
    label = config['classes'][int(np.argmax(sample[1]))]
    ext_spec = spec[np.newaxis, :]
    heatmap = make_gradcam_heatmap(ext_spec, best_model, last_conv_layer_name,
                                   pred_index=pred_idx)
    resized_heatmap = resize(heatmap, (spec.shape[0], spec.shape[1]),
                             anti_aliasing=True)
    heatmap_lst.append(resized_heatmap)
    spectrogram_lst.append(spec)
    label_lst.append(label)

# VISUALIZE SPECTROGRAMS
x_ticks_labels = np.array([0, 0.25, 0.5, 0.75, 1])
x_ticks = (spec.shape[1]*x_ticks_labels).astype(int)
y_ticks_labels = np.linspace(4000, config['sr']/2, num=9)
y_ticks = ((y_ticks_labels-4000)*config['n_fft']/config['sr']).astype(int)
fig = plt.figure(figsize=(10, 16))
fig.suptitle('Spectrograms', fontsize=16)
for idx, (label, spec, heat) in enumerate(zip(label_lst, spectrogram_lst,
                                              heatmap_lst)):
    plt.subplot(3, 3, idx+1)
    plt.title(f"{label}", fontsize=16)
    plt.imshow(spec, cmap='gray', interpolation='none', origin='lower')
    plt.imshow(heat, alpha=0.5, cmap='hot', interpolation='none',
               origin='lower')
    plt.xticks(x_ticks, x_ticks_labels)
    plt.yticks(y_ticks, y_ticks_labels)
    plt.tight_layout()
# %% Mean heatmap


heatmap_lst = []
label_lst = []
heatmap_arr = np.empty((35, 16, 100))
for label_idx in range(0, len(ds_list), 100):
    # Calc mean heatmap
    for idx in range(0, 100, 1):
        sample = ds_list[idx+label_idx]
        spec = sample[0]
        ext_spec = spec[np.newaxis, :]
        heatmap_arr[:,:,idx] = make_gradcam_heatmap(ext_spec, best_model,
                                                    last_conv_layer_name,
                                                    pred_index=pred_idx)

    heatmap_mean = np.mean(heatmap_arr, axis=2)
    resized_heatmap = resize(heatmap_mean, (spec.shape[0], spec.shape[1]),
                                anti_aliasing=True)
    heatmap_lst.append(resized_heatmap)
    # Get label
    label_lst.append(config['classes'][int(np.argmax(sample[1]))])

fig = plt.figure(figsize=(10, 16))
fig.suptitle('Mean Heatmaps', fontsize=16)
for idx, (label, heat) in enumerate(zip(label_lst, heatmap_lst)):
    plt.subplot(3, 3, idx+1)
    plt.title(f"{label}", fontsize=16)
    plt.imshow(heat, cmap='hot', interpolation='none', origin='lower')
    plt.xticks(x_ticks, x_ticks_labels)
    plt.yticks(y_ticks, y_ticks_labels)
    plt.tight_layout()
# %%
