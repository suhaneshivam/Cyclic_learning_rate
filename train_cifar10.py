import matplotlib
matplotlib.use("Agg")

from helper.clr_callback import CyclicLR
from helper.minigooglenet import MiniGoogLeNet
from helper import config
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

((trainX ,trainY) ,(testX ,testY)) = cifar10.load_data()

trainX = trainX.astype("float32")
testX = testX.astype("float32")

mean_train = np.mean(trainX ,axis = 0)

trainX = trainX - mean_train
testX = testX - mean_train

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1 ,height_shift_range=0.1 ,horizontal_flip=True ,fill_mode="nearest")

opt = SGD(learning_rate = config.MIN_LR ,momentum = 0.9)
model = MiniGoogLeNet.build(width = 32 ,height = 32 , depth = 3 ,classes = 10)
model.compile(loss = "categorical_crossentropy" ,optimizer= opt ,metrics = ["accuracy"])

print("[INFO] using '{}' method".format(config.CLR_METHOD))
clr = CyclicLR(base_lr = config.MIN_LR ,max_lr = config.MAX_LR ,
                        step_size = config.STEP_SIZE*(trainX.shape[0]//config.BATCH_SIZE) ,
                            mode = config.CLR_METHOD )

callbacks = [clr]

H = model.fit(x = aug.flow(trainX ,trainY ,batch_size=config.BATCH_SIZE) ,
                            validation_data=(testX ,testY) ,steps_per_epoch=trainX.shape[0] //config.BATCH_SIZE ,
                            epochs = config.NUM_EPOCHS ,
                            verbose=True ,callbacks=callbacks)
predictions = model.predict(testX ,batch_size = config.BATCH_SIZE)
print(classification_report(testY.argmax(axis = 1) ,predictions.argmax(axis = 1) ,target_names=config.CLASSES))

N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)
# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)
