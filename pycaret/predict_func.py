import argparse
import csv
import time
import os
import numpy as np
import pandas as pd
import pycaret.regression as pcr
import sklearn.metrics as sklm
from support.regressor_support import \
        read_csv_dataset

def compute_measures(prediction):
    for measure in args.measures:
        exp_measure = 'lambda _ : sklm.' + measure
        value_of_measure = eval(exp_measure)(None)(prediction[args.target_column], prediction['Label'])
        print("%s: %.8f" % (measure, value_of_measure))

def save_prediction(head, prediction):
    csv_output_file = open(args.prediction_data_file, 'w')
    label_index = prediction.columns.get_loc('Label')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        writer.writerow(head)
        for i in range(0, len(prediction)):
            row = []
            for j in range(0, len(prediction.columns)):
                if prediction.columns[j] != 'Label' and prediction.columns[j] != args.target_column:
                    row.append(prediction.iat[i, j])
            row.append(prediction.iat[i, label_index])
            writer.writerow(row)
    print("Generated one-variable function predicted data '%s'" % args.prediction_data_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s makes prediction of the values of a multiple-input single-output function with the best pretrained regressor chosen by PyCaret')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--model',
                        type=str,
                        dest='model_file',
                        required=True,
                        help='model file')

    parser.add_argument('--ds',
                        type=str,
                        dest='df_prediction',
                        required=True,
                        help='dataset file (csv format)')

    parser.add_argument('--targetcol',
                        type=str,
                        dest='target_column',
                        required=True,
                        help='Target column name')

    parser.add_argument('--predictionout',
                        type=str,
                        dest='prediction_data_file',
                        required=True,
                        help='prediction data file (csv format)')

    parser.add_argument('--measures',
                        type=str,
                        dest='measures',
                        required=False,
                        nargs = '+',
                        default = [],
                        help='List of built-in sklearn regression measures to compare prediction with input dataset')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    df_prediction = read_csv_dataset(args.df_prediction)
    columns = df_prediction.columns.tolist()

    final_model = pcr.load_model(args.model_file)

    start_time = time.time()
    prediction = pcr.predict_model(final_model, data=df_prediction)
    elapsed_time = time.time() - start_time
    print ("Predicting time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    compute_measures(prediction)
    save_prediction(columns, prediction)

    print("#### Terminated %s ####" % os.path.basename(__file__));
