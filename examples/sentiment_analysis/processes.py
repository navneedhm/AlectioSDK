import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from model import RNNModel
from tqdm import tqdm

if not os.path.isdir('./.data'):
    os.mkdir('./.data')

dataset, info = tfds.load('imdb_reviews/subwords8k', data_dir="./data", as_supervised=True, with_info=True)
# dataset = tfds.as_numpy(dataset)
train_dataset, test_dataset = dataset['train'], dataset['test']

# encoder = info.features['text'].encoder
BUFFER_SIZE = 10000


if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("Using GPU")
else:
    print("Using CPU")

# TODO: use tf.Dataset and get filtering to work
train_dset = list(train_dataset.as_numpy_iterator())
test_dset = list(test_dataset.as_numpy_iterator())


def getdatasetstate(args={}):
    return {k: k for k in range(25000)}

def train(args, labeled, resume_from, ckpt_file):

    rnn_model = RNNModel()

    batch_size = args["batch_size"]
    epochs = args["train_epochs"]

    labeled_train1 = [train_dset[i] for i in labeled]
    labeled_train = tf.data.Dataset.from_generator(lambda:labeled_train1, (tf.float32, tf.float32), (tf.TensorShape([None]), tf.TensorShape([]))).padded_batch(batch_size)

    if resume_from is not None:
        # ckpt = tf.keras.models.load_model(os.path.join(args["EXPT_DIR"], resume_from))
        rnn_model = tf.keras.models.load_model(os.path.join(args["EXPT_DIR"], resume_from))

    # history = rnn_model.fit(labeled_train, epochs=epochs)
    loss_func=tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer=tf.keras.optimizers.Adam(1e-4)

    train_loss_avg = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.BinaryAccuracy()

    for epoch in range(epochs):
        train_loss_avg.reset_states()
        train_acc.reset_states()
        for data, labels in tqdm(labeled_train, desc="Training"):
            with tf.GradientTape() as tape:
                pred = rnn_model(data)
                # print(pred)
                loss = loss_func(pred, labels)

                train_loss_avg.update_state(loss)
                train_acc.update_state(labels, rnn_model(data))

                gradients = tape.gradient(loss, rnn_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))

        print("Training Accuracy: {}".format(train_acc.result().numpy()))
        print("Training Loss: {}".format(train_loss_avg.result().numpy()))


    print("Finished Training. Saving the model as {}".format(ckpt_file))
    tf.keras.models.save_model(rnn_model,  os.path.join(args["EXPT_DIR"], ckpt_file), save_format = "h5")

    return

def test(args, ckpt_file):

    batch_size = args["batch_size"]
    labeled_test = tf.data.Dataset.from_generator(lambda:test_dset, (tf.float32, tf.float32), (tf.TensorShape([None]), tf.TensorShape([]))).padded_batch(batch_size)

    predictions, targets = [], []
    rnn_model = tf.keras.models.load_model(os.path.join(args["EXPT_DIR"], ckpt_file))

    for data, labels in tqdm(labeled_test, desc="Testing"):
        pred = rnn_model(data).numpy().flatten()
        pred[pred <= 0] = 0
        pred[pred > 0] = 1
        predictions.extend(pred.tolist())
        targets.extend(labels.numpy().tolist())

    print("Testing Accuracy : {}".format(1 - (np.sum(np.abs(np.array(predictions) - np.array(targets)))/len(predictions))))

    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    batch_size = args["batch_size"]
    epochs = args["train_epochs"]
    unlabeled_train1 = [train_dset[i] for i in unlabeled]
    unlabeled_train = tf.data.Dataset.from_generator(lambda:unlabeled_train1, (tf.float32, tf.float32), (tf.TensorShape([None]), tf.TensorShape([]))).padded_batch(batch_size)

    rnn_model = tf.keras.models.load_model(os.path.join(args["EXPT_DIR"], ckpt_file))
    outputs_fin = {}
    i = 0
    for data, labels in tqdm(unlabeled_train, desc="Inferring"):
        outputs = rnn_model(data).numpy().flatten()
        pred = np.copy(outputs)
        pred[pred <= 0] = 0
        pred[pred > 0] = 1

        for j in range(len(pred)):
            outputs_fin[i] = {}
            outputs_fin[i]["prediction"] = pred[j]
            outputs_fin[i]["pre_softmax"] = outputs[j]
            i += 1


    return {"outputs": outputs_fin}

if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    # test(ckpt_file=ckpt_file)
    # infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
