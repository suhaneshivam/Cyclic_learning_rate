import os

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
	"frog", "horse", "ship", "truck"]

MIN_LR = 1e-7
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 96

TRAINING_PLOT_PATH = os.path.sep.join(["output" ,"training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output" ,"clr_plot.png"])
