import argparse
import numpy as np
from sklearn.metrics import classification_report
import pickle
from mlclass import BinaryLogisticRegression, MultiLabelsClassifier
from mlclass import load_data, get_labeled_data, scale_data_and_add_bayes

TEST_DATA_FILENAME = 't10k-images-idx3-ubyte'
TEST_LABEL_FILENAME = 't10k-labels-idx1-ubyte'
MODEL_FILE_NAME = 'baranov_model.pickle'


def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_test_dir')
    parser.add_argument('-y_test_dir')
    parser.add_argument('-model_input_dir')
    return parser


def check_args(args):

    if args.x_test_dir[-1] != '/':
        x_test_dir = args.x_test_dir + '/'
    else:
        x_test_dir = args.x_test_dir

    if args.y_test_dir[-1] != '/':
        y_test_dir = args.y_test_dir + '/'
    else:
        y_test_dir = args.y_test_dir

    if args.model_input_dir[-1] != '/':
        model_input_dir = args.model_input_dir + '/'
    else:
        model_input_dir = args.model_input_dir

    return x_test_dir, y_test_dir, model_input_dir


def get_model(path):
    with open(path+MODEL_FILE_NAME, 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    args_parser = create_args_parser()
    args = args_parser.parse_args()
    x_test_dir, y_test_dir, model_input_dir = check_args(args)
    x_data, y_data = load_data(x_test_dir + TEST_DATA_FILENAME, y_test_dir + TEST_LABEL_FILENAME)
    x_data = scale_data_and_add_bayes(x_data)
    model = get_model(model_input_dir)
    y_pred = model.predict(x_data)
    print(classification_report(y_data.astype('int'), y_pred.astype('int')))


if __name__ == '__main__':
    main()