import argparse
import numpy as np
from sklearn.metrics import classification_report
import pickle
from mlclass import MultiLabelsClassifier
from mlclass import load_data, get_labeled_data, scale_data_and_add_bayes

TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME = 'train-labels-idx1-ubyte'
MODEL_FILE_NAME = 'baranov_model.pickle'


def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_train_dir')
    parser.add_argument('-y_train_dir')
    parser.add_argument('-model_output_dir')
    return parser


def check_args(args):

    if args.x_train_dir[-1] != '/':
        x_train_dir = args.x_train_dir + '/'
    else:
        x_train_dir = args.x_train_dir

    if args.y_train_dir[-1] != '/':
        y_train_dir = args.y_train_dir + '/'
    else:
        y_train_dir = args.y_train_dir

    if args.model_output_dir[-1] != '/':
        model_output_dir = args.model_output_dir + '/'
    else:
        model_output_dir = args.model_output_dir

    return x_train_dir, y_train_dir, model_output_dir


def save_model(model):
    with open(MODEL_FILE_NAME, 'wb') as f:
        pickle.dump(model, f)


def main():
    args_parser = create_args_parser()
    args = args_parser.parse_args()
    x_train_dir, y_train_dir, model_output_dir = check_args(args)
    x_data, y_data = load_data(x_train_dir + TRAIN_DATA_FILENAME, y_train_dir + TRAIN_LABEL_FILENAME)
    x_data = scale_data_and_add_bayes(x_data)
    model = MultiLabelsClassifier()
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)
    print(classification_report(y_data.astype('int'), y_pred.astype('int')))
    save_model(model)


if __name__ == '__main__':
    main()