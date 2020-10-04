import argparse
import csv
import time
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib as jl
import sklearn.metrics as sklm
from support.regressor_support import add_args_for_regressor, prepare_kwargs_for_regressor, read_csv_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s fits a multiple-input single-output (scalar) function dataset using a configurable XGBoost')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='train dataset file (csv format)')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_file',
                        required=True,
                        help='output model file')

    parser.add_argument('--valds',
                        type=str,
                        dest='val_dataset_filename',
                        required=False,
                        help='validation dataset file (csv format)')

    add_args_for_regressor(parser)

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    head_train, df_independent_train, df_dependent_train = read_csv_dataset(args.train_dataset_filename, 1)
    if args.val_dataset_filename is not None:
        head_val, df_independent_val, df_dependent_val = read_csv_dataset(args.val_dataset_filename, 1)

    xgb_kwargs = prepare_kwargs_for_regressor(args)
    model = xgb.XGBRegressor(**xgb_kwargs)

    start_time = time.time()
    model.fit(df_independent_train, df_dependent_train)
    elapsed_time = time.time() - start_time
    print ("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    jl.dump(model, args.model_file)
    print("Generated one-variable function xgboost model '%s'" % args.model_file)

    print("#### Terminated %s ####" % os.path.basename(__file__));
