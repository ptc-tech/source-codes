# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-01-28 10:56:25
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-01-28 12:44:38

# import the necessary packages
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
# import cv2
import matplotlib
matplotlib.use("macosx")

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", type=str, required=True,
#     help="path to input image dataset file")
# ap.add_argument("-m", "--model", type=str, required=True,
#     help="path to trained autoencoder")
# ap.add_argument("-q", "--quantile", type=float, default=0.999,
#     help="q-th quantile used to identify outliers")
# args = vars(ap.parse_args())

args = {"model": "autoencoder.model",
    "dataset": "images.pickle",
    "quantile": 0.999}


# load the model and image data from disk
print("[INFO] loading autoencoder and image data...")
autoencoder = load_model(args["model"])
# images = pickle.loads(open(args["dataset"], "rb").read())


((trainX, trainY), (testX, testY)) = mnist.load_data()

# build our unsupervised dataset of images with a small amount of
# contamination (i.e., anomalies) added into it
print("[INFO] creating unsupervised dataset...")
# images = build_unsupervised_dataset(trainX, trainY, valid_label=1,
#     anomaly_label=3, contam=0.05)
# images = np.where(trainY == 1)[0]
baseline_images = trainX[np.where(trainY == 1)[0]]
baseline_images = np.expand_dims(baseline_images, axis=-1)
baseline_images = baseline_images.astype("float32") / 255.0


# make predictions on our image data and initialize our list of
# reconstruction errors
decoded = autoencoder.predict(baseline_images)

# loop over all original images and their corresponding
# reconstructions
errors = []
for (image, reconstructed) in zip(baseline_images, decoded):
    # compute the mean squared error between the ground-truth image
    # and the reconstructed image, then add it to our list of errors
    mse = np.mean((image - reconstructed) ** 2)
    errors.append(mse)


# compute the q-th quantile of the errors which serves as our
# threshold to identify anomalies -- any data point that our model
# reconstructed with > threshold error will be marked as an outlier
thresh = np.quantile(errors, args["quantile"])
idxs = np.where(np.array(errors) >= thresh)[0]
print("[INFO] mse threshold: {}".format(thresh))
print("[INFO] {} outliers found".format(len(idxs)))


# =======

images_to_analyse = trainX[np.where(trainY == 3)[0]]
images_to_analyse = images_to_analyse[:100]
images_to_analyse = np.expand_dims(images_to_analyse, axis=-1)
images_to_analyse = images_to_analyse.astype("float32") / 255.0


decoded = autoencoder.predict(images_to_analyse)

# loop over all original images and their corresponding
# reconstructions
errors = []
for (image, reconstructed) in zip(images_to_analyse, decoded):
    # compute the mean squared error between the ground-truth image
    # and the reconstructed image, then add it to our list of errors
    mse = np.mean((image - reconstructed) ** 2)
    errors.append(mse)

# ======

images_to_analyse = trainX[np.where(trainY == 3)[0]]
images_to_analyse = images_to_analyse[:100]
images_to_analyse = np.vstack([images_to_analyse, trainX[np.where(trainY == 1)[0]][:100]])
images_to_analyse = np.expand_dims(images_to_analyse, axis=-1)
images_to_analyse = images_to_analyse.astype("float32") / 255.0


decoded = autoencoder.predict(images_to_analyse)

# loop over all original images and their corresponding
# reconstructions
errors = []
for (image, reconstructed) in zip(images_to_analyse, decoded):
    # compute the mean squared error between the ground-truth image
    # and the reconstructed image, then add it to our list of errors
    mse = np.mean((image - reconstructed) ** 2)
    errors.append(mse)



idxs = np.where(np.array(errors) >= thresh)[0]

# initialize the outputs array
outputs = None
random_idxs = np.random.randint(0, len(idxs), np.min([len(idxs), 20]))
# loop over the indexes of images with a high mean squared error term
# for i in idxs[50:60]:
for i in random_idxs:
    # grab the original image and reconstructed image
    original = (images_to_analyse[i] * 255).astype("uint8")
    reconstructed = (decoded[i] * 255).astype("uint8")
    # stack the original and reconstructed image side-by-side
    output = np.hstack([original, reconstructed])
    # if the outputs array is empty, initialize it as the current
    # side-by-side image display
    if outputs is None:
        outputs = output
    # otherwise, vertically stack the outputs
    else:
        outputs = np.vstack([outputs, output])


plt.imshow(outputs)
plt.show()

# show the output visualization
# cv2.imshow("Output", outputs)
# cv2.waitKey(0)




#######################################################################################################################

                                         # === END OF FILE === #

#######################################################################################################################
