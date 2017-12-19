import mindwave
import time
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

"""
Req: numpy, matplotlib, scikit-learn

Step 0. Run python script(train)
 $ python test_viz.py train

Step 1. Connect Headset
 Find MAC address(XX:XX:XX:XX:XX:XX) of MindWave Mobile
 $ hcitool scan
 Connect to MindWave Mobile
 $ sudo rfcomm connect /dev/rfcomm0 MAC_ADDRESS
 Check connection(on another terminal)
 $ hexdump /dev/rfcomm0

Step 2. Run python script(test)
 Sample eeg acquisition code using
 https://github.com/BarkleyUS/mindwave-python
 $ python test_viz.py test

cf. Available variables
 poor_signal, status,
 attention, meditation,
 blink, raw_value
"""

TRAIN_SAMPLE = 500
HISTORY = 100
THRESHOLD = 50
MODEL_FILENAME = "lin_clf.sav"

# Simulate data stream
class Manus:
    def __init__(self):
        self.n = 0
        self.n_max = HISTORY
        self.feature = 0
    def simulate(self):
        self.feature = (np.sin(self.n) + np.random.random_sample()) * 50
        self.n = self.n + 1
        if self.n >=  self.n_max:
            self.n = 0
        return self.feature

# Classify EEG feature into 2 classes
def classify(model, feature, ml=True):
    if ml != True:
	if feature >= THRESHOLD:
            feature = 1
        else:
            feature = 0
        return feature
    else:
        feature = model.predict(feature)
        return feature

# Print messages based on predictions
def action(prediction):
    if prediction == 1:
        print "True"
    elif prediction == 0:
        print "False"
    else:
        print "Non proper prediction"

def train(mode):
    if mode == "raw":
        # Connect headset
        headset = mindwave.Headset('/dev/rfcomm0')
        time.sleep(2)
        headset.connect()
    else:
        manus = Manus()

    X = np.zeros((TRAIN_SAMPLE, HISTORY))
    Y = np.zeros(TRAIN_SAMPLE)
    print "Acquiring True condition..."
    for i in range(TRAIN_SAMPLE/2):
        for k in range(HISTORY):
            if mode == "raw":
                X[i, k] = headset.attention
            else:
                X[i, k] = manus.simulate()
        if i%10 == 0:
            print i
    Y[:TRAIN_SAMPLE/2-1] = 1

    print "Acquiring False condition..."
    for i in range(TRAIN_SAMPLE/2):
        for k in range(HISTORY):
            if mode == "raw":
                X[i+TRAIN_SAMPLE/2, k] = headset.attention
            else:
                X[i+TRAIN_SAMPLE/2, k] = manus.simulate()
        if i%10 == 0:
            print i
    Y[TRAIN_SAMPLE/2:] = 0

    print "Start training."
    lin_clf = LinearSVC()
    lin_clf.fit(X, Y)
    print "Training finished."
    pickle.dump(lin_clf, open(MODEL_FILENAME, 'wb'))

def test(mode):
    if mode == "raw":
        # Connect headset
        headset = mindwave.Headset('/dev/rfcomm0')
        time.sleep(2)
        headset.connect()
    else:
        manus = Manus()

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2,1)
    x = np.arange(0, HISTORY, 1)
    y = np.zeros(HISTORY)

    lines1, = ax1.plot(x, y)
    ax1.set_title('Attention')
    ax1.set_xlabel('t')
    ax1.set_xlim((x.min(), x.max()))
    ax1.set_ylim((-1, 101))
    lines2, = ax2.plot(x, y)
    ax2.set_title('Prediction')
    ax2.set_xlabel('t')
    ax2.set_xlim((x.min(), x.max()))
    ax2.set_ylim((-0.1, 1.1))
    plt.tight_layout()

    # Load model
    model = pickle.load(open(MODEL_FILENAME, 'rb'))

    n = 0
    feature = 0
    pred = 0
    feat_seq = np.zeros(HISTORY)
    pred_seq = np.zeros(HISTORY)

    try:
        while True:
            # Read value
            if mode == "raw":
                feature = headset.attention
            else:
                feature = manus.simulate()
            feat_seq[n] = feature

            # Classify
            prediction = classify(model, feat_seq.reshape(1, -1), ml=True)
            pred_seq[n] = prediction
            action(prediction)

            # Plot
            lines1.set_data(x, feat_seq)
            lines2.set_data(x, pred_seq)
            ax2.set_xlim((0, 100))
            plt.pause(0.01)

            n = n + 1
            if n >= HISTORY:
                n = 0

    except KeyboardInterrupt:
        print "Keyboard interruption. Halting."

if __name__ == '__main__':
    argv = sys.argv

    if (len(argv) != 3):
        print "Usage: $ python %s [train/test] [raw/sim]" % argv[0]
    elif argv[1] == "train":
        train(argv[2])
    elif argv[1] == "test":
        test(argv[2])
    else:
        print "Invalid command."
