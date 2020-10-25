import argparse
import csv
import time
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib as jl
import sklearn.metrics as sklm
from support.regressor_support import read_csv_dataset

def compute_measures(df_dependent, prediction):
    for measure in args.measures:
        exp_measure = 'lambda _ : sklm.' + measure
        value_of_measure = eval(exp_measure)(None)(df_dependent, prediction)
        print("%s: %.8f" % (measure, value_of_measure))

def save_prediction(df_independent, prediction):
    csv_output_file = open(args.prediction_data_file, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        writer.writerow(head)
        for i in range(0, len(df_independent)):
            row = []
            for j in range(0, len(df_independent.columns)):
                row.append(df_independent.iat[i, j])
            for p in prediction[i]:
                row.append(p)
            writer.writerow(row)
    print("Generated one-variable function predicted data '%s'" % args.prediction_data_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s makes prediction of the values of a multiple-input single-output (scalar) function with a pretrained XGBoost model')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.1')

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

    parser.add_argument('--outputdim',
                        type=int,
                        dest='num_of_dependent_columns',
                        required=True,
                        help='Output dimension (alias the number of dependent columns, that must be last columns)')

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
                        help='List of built-in sklearn regression metrics to compare prediction with input dataset')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    head, df_independent, df_dependent = read_csv_dataset(args.df_prediction, args.num_of_dependent_columns)

    model = jl.load(args.model_file)

    start_time = time.time()
    prediction = model.predict(df_independent)
    elapsed_time = time.time() - start_time
    print ("Predicting time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    compute_measures(df_dependent, prediction)
    save_prediction(df_independent, prediction)

    print("#### Terminated %s ####" % os.path.basename(__file__));
