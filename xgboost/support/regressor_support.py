import pandas as pd

def add_args_for_regressor(parser):

    parser.add_argument('--xgbparams',
                        type=str,
                        dest='xgb_params',
                        required=False,
                        help='parameters of XGBoost constructor')

def prepare_kwargs_for_regressor(args):
    if args.xgb_params != None:
        xgb_kwargs = eval('{' + args.xgb_params + '}')
    else:
        xgb_kwargs = {}
    return xgb_kwargs

def read_csv_dataset(dsfilename, num_of_dependent_columns):
    df = pd.read_csv(dsfilename, sep=',', header=0)
    columns = df.columns.tolist()
    cols_independent = columns[:len(columns) - num_of_dependent_columns]
    cols_dependent = columns[len(columns) - num_of_dependent_columns:]
    df_independent = df[cols_independent]
    df_dependent = df[cols_dependent]
    return df.head(), df_independent, df_dependent
