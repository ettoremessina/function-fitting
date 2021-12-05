import argparse
import csv
import time
import os
import numpy as np
import pandas as pd
import pycaret.regression as pcr
import warnings

from support.regressor_support import \
	read_csv_dataset, \
	retrieve_independent_column_names, \
	prepare_kwargs_for_setup, \
	prepare_kwargs_for_compare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s fits a multiple-input single-output function dataset using the best regressor chosen by PyCaret')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='Train dataset file (csv format)')

    parser.add_argument('--targetcol',
                        type=str,
                        dest='target_column',
                        required=True,
                        help='Target column name')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_file',
                        required=True,
                        help='Output model file')

    parser.add_argument('--metric',
                        type=str,
                        dest='metric',
                        required=False,
                        default='R2',
                        help='metric to evaluate the best model')

    parser.add_argument('--setupparams',
                        type=str,
                        dest='setup_params',
                        required=False,
                        help='Parameters of PyCaret regression.setup function')

    parser.add_argument('--compareparams',
                        type=str,
                        dest='compare_params',
                        required=False,
                        help='Parameters of PyCaret regression.compare_models function')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

    df_train = read_csv_dataset(args.train_dataset_filename)

    setup_kwargs = prepare_kwargs_for_setup(args)
    exp_regression = pcr.setup(
        data=df_train,
        target=args.target_column,
        numeric_features = retrieve_independent_column_names(df_train, args.target_column),
        silent=True,
        html=False,
        **setup_kwargs)

    start_time = time.time()

    compare_kwargs = prepare_kwargs_for_compare(args)
    top_model = pcr.compare_models(n_select = 1, sort=args.metric, **compare_kwargs)
    if isinstance(top_model, list):
        top_model = top_model[0]
    tuned_model = pcr.tune_model(top_model, optimize=args.metric)
    final_model = pcr.finalize_model(tuned_model)

    elapsed_time = time.time() - start_time
    print ("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    pcr.save_model(final_model, args.model_file)
    print("Generated one-variable function '%s' Regressor model '%s'" %
        (final_model.__class__.__name__, args.model_file))

    print("#### Terminated %s ####" % os.path.basename(__file__));
