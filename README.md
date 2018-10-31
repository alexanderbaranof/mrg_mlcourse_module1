# mrg_mlcourse_module1

Для запуска кода использовать

- python3 train.py -x_train_dir=./ -y_train_dir=./ -model_output_dir=./

- python3 predict.py -x_test_dir=./ -y_test_dir=./ -model_input_dir=./


Использованный подход: логистическая регрессия one-vs-all 

Из-за использованного подхода обучается сравнительно долго т.к. градиент вычисляется полностью.
