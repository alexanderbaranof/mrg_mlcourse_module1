import numpy as np
from struct import unpack
from numpy import zeros, uint8, float32

NUM_EPOCHS = 1000
MAX_LEARNING_RATE = 6.4


class BinaryLogisticRegression:
    def __init__(self, n_epochs=NUM_EPOCHS):
        self._weights = None
        self._n_epochs = n_epochs
        self._learning_rate = 0.1

    def _loss_func(self, x, y):
        n = x.shape[0]
        p = self._sigmoid(np.sum(x * self._weights, axis=1))
        o = np.ones((1, x.shape[0]))[0]
        res = -1 / n * np.sum(y * np.log(p) + (o - y) * np.log(o - p))
        w = np.copy(self._weights)
        w[0] = 0
        return res + 0.0001 / 2 * 1 / n * np.sum(w ** 2)

    def _grad_step(self, x, y):
        n = x.shape[0]
        p = np.sum(x * self._weights, axis=1)
        p = self._sigmoid(p)
        p_y = p - y
        grad = np.sum(x.T * p_y, axis=1)
        grad /= n
        w = np.copy(self._weights)
        w[0] = 0  # не применяем l2 к баесу
        grad += 0.0001 * w
        return grad

    def fit(self, x, y):
        self._weights = np.random.uniform(-0.5, 0.5, [1, x.shape[1]])[0]
        prev_loss = np.inf
        for epoch in range(self._n_epochs):
            grad = self._learning_rate * self._grad_step(x, y)
            self._weights -= grad
            loss = self._loss_func(x, y)
            if loss <= prev_loss:
                self._learning_rate *= 2
                self._learning_rate = min(MAX_LEARNING_RATE, self._learning_rate)
            else:
                self._learning_rate /= 4
            prev_loss = loss

    def _sigmoid(self, tau):
        tau = np.array(tau, dtype=np.float128)
        return 1 / (1 + np.exp(-tau))

    def predict_proba(self, x):
        p = np.sum(x * self._weights, axis=1)
        p = self._sigmoid(p)
        return p


class MultiLabelsClassifier:
    def __init__(self):
        self._models = None
        self._number_of_labels = None

    def fit(self, x, y):
        labels = set(y)
        self._number_of_labels = len(labels)
        models = [BinaryLogisticRegression() for _ in range(len(labels))]
        self._models = dict(zip(labels, models))
        for i in self._models:
            tmp_y = np.copy(y)
            tmp_y[tmp_y != i] = -1
            tmp_y[tmp_y == i] = 1
            tmp_y[tmp_y == -1] = 0
            self._models[i].fit(x, tmp_y)

    def _sigmoid(self, tau):
        tau = np.array(tau)
        return 1 / (1 + np.exp(-tau))

    def predict(self, x):

        result_matrix = None

        for i in range(self._number_of_labels):
            p = self._models[i].predict_proba(x)

            if result_matrix is None:
                result_matrix = p.T[:, np.newaxis]
            else:
                result_matrix = np.hstack((result_matrix, p.T[:, np.newaxis]))

        y_pred = np.array([])
        for i in range(result_matrix.shape[0]):
            number = np.where(result_matrix[i] == np.max(result_matrix[i]))[0][0]
            y_pred = np.append(y_pred, number)

        return y_pred



def load_data(x_file_path, y_file_path):
    x,y = get_labeled_data(x_file_path, y_file_path)
    return x, y


def get_labeled_data(imagefile, labelfile):
    """Частично эту функцию я взял отсюда https://martin-thoma.com/classify-mnist-with-pybrain/"""
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows*cols), dtype=float32)  # Initialize numpy array
    y = np.array([], dtype=uint8)  # Initialize numpy array
    for i in range(N):
        tmp_image = zeros((rows*cols), dtype=float32)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                tmp_image[row*cols + col] = tmp_pixel
        x[i] = tmp_image
        tmp_label = labels.read(1)
        y = np.append(y, unpack('>B', tmp_label)[0])
    return x, y


def scale_data_and_add_bayes(x):
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    vec_max = x.max(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.true_divide(x, vec_max)
        x[x == np.inf] = 0
        x = np.nan_to_num(x)

    return x
