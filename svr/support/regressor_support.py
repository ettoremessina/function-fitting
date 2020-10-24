import pandas as pd

def read_csv_dataset(dsfilename, num_of_dependent_columns):
    df = pd.read_csv(dsfilename, sep=',', header=0)
    columns = df.columns.tolist()
    cols_independent = columns[:len(columns) - num_of_dependent_columns]
    cols_dependent = columns[len(columns) - num_of_dependent_columns:]
    df_independent = df[cols_independent]
    df_dependent = df[cols_dependent]
    return df.head(), df_independent, df_dependent

def prepare_kwargs_for_regressor(args):
    if args.svr_params != None:
        svr_kwargs = eval('{' + args.svr_params + '}')
    else:
        svr_kwargs = {}
    return svr_kwargs
