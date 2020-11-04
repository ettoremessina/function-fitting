import argparse
import csv
import time
import os
import numpy as np
import pandas as pd
import joblib as jl
import sklearn.svm as sklsvm
import sklearn.metrics as sklm
import sklearn.multioutput as sklmo
from support.regressor_support import read_csv_dataset, prepare_kwargs_for_regressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s fits a multiple-input multiple-output function dataset using a configurable Epsilon-Support Vector Regressor')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='Train dataset file (csv format)')

    parser.add_argument('--outputdim',
                        type=int,
                        dest='num_of_dependent_columns',
                        required=True,
                        help='Output dimension (alias the number of dependent columns, that must be last columns)')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_file',
                        required=True,
                        help='Output model file')

    parser.add_argument('--svrparams',
                        type=str,
                        dest='svr_params',
                        required=False,
                        help='Parameters of Epsilon-Support Vector Regressor constructor')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    head_train, df_independent_train, df_dependent_train = read_csv_dataset(args.train_dataset_filename, args.num_of_dependent_columns)

    svr_kwargs = prepare_kwargs_for_regressor(args)
    model = sklmo.MultiOutputRegressor(sklsvm.SVR(**svr_kwargs))

    start_time = time.time()
    model.fit(
        df_independent_train,
        df_dependent_train)
    elapsed_time = time.time() - start_time
    print ("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    jl.dump(model, args.model_file)
    print("Generated one-variable function Epsilon-Support Vector Regressor model '%s'" % args.model_file)

    print("#### Terminated %s ####" % os.path.basename(__file__));
