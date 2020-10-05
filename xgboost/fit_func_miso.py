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

def prepare_kwargs_for_regressor(args):
    if args.xgb_params != None:
        xgb_kwargs = eval('{' + args.xgb_params + '}')
    else:
        xgb_kwargs = {}
    return xgb_kwargs

def save_dumps(results):
    if args.dumpout_path is not None:
        if not os.path.exists(args.dumpout_path):
            os.makedirs(args.dumpout_path)
        for metric in args.val_metrics:
            dump_fn = os.path.join(args.dumpout_path, 'metric_' + metric + '.csv')
            np.savetxt(dump_fn, results['validation_0'][metric], delimiter=',')
            print("Generated dump file '%s'" % dump_fn)
            if (len(results.keys()) > 1):
                dump_fn = os.path.join(args.dumpout_path, 'val_' + metric + '.csv')
                np.savetxt(dump_fn, results['validation_1'][metric], delimiter=',')
                print("Generated dump file '%s'" % dump_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s fits a multiple-input single-output (scalar) function dataset using a configurable XGBoost')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='Train dataset file (csv format)')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_file',
                        required=True,
                        help='Output model file')

    parser.add_argument('--valds',
                        type=str,
                        dest='val_dataset_filename',
                        required=False,
                        help='Validation dataset file (csv format)')

    parser.add_argument('--metrics',
                        type=str,
                        dest='val_metrics',
                        required=False,
                        nargs = '+',
                        default = [],
                        help='List of built-in evaluation metrics to apply to validation dataset')

    parser.add_argument('--dumpout',
                        type=str,
                        dest='dumpout_path',
                        required=False,
                        help='Dump directory (directory to store metric values)')

    parser.add_argument('--earlystop',
                        type=int,
                        dest='early_stopping_rounds',
                        required=False,
                        help='Number of round for early stopping')

    parser.add_argument('--xgbparams',
                        type=str,
                        dest='xgb_params',
                        required=False,
                        help='Parameters of XGBoost constructor')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    head_train, df_independent_train, df_dependent_train = read_csv_dataset(args.train_dataset_filename, 1)
    eval_set = [(df_independent_train, df_dependent_train)]

    if args.val_dataset_filename is not None:
        head_val, df_independent_val, df_dependent_val = read_csv_dataset(args.val_dataset_filename, 1)
        eval_set.append((df_independent_val, df_dependent_val))

    xgb_kwargs = prepare_kwargs_for_regressor(args)
    model = xgb.XGBRegressor(**xgb_kwargs)

    start_time = time.time()
    model.fit(
        df_independent_train,
        df_dependent_train,
        eval_set=eval_set,
        eval_metric=args.val_metrics,
        early_stopping_rounds=args.early_stopping_rounds)
    elapsed_time = time.time() - start_time
    print ("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    results = model.evals_result()

    jl.dump(model, args.model_file)
    print("Generated one-variable function xgboost model '%s'" % args.model_file)

    save_dumps(results)

    print("#### Terminated %s ####" % os.path.basename(__file__));
