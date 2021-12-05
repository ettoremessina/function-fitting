import pandas as pd

def read_csv_dataset(dsfilename):
    df = pd.read_csv(dsfilename, sep=',', header=0)
    return df

def retrieve_independent_column_names(df, target):
    columns = df.columns.tolist()
    cols_independent_names = [column for column in columns if column != target]
    return cols_independent_names

def prepare_kwargs_for_setup(args):
    if args.setup_params != None:
        setup_kwargs = eval('{' + args.setup_params + '}')
    else:
        setup_kwargs = {}
    return setup_kwargs

def prepare_kwargs_for_compare(args):
    if args.compare_params != None:
        compare_kwargs = eval('{' + args.compare_params + '}')
    else:
        compare_kwargs = {}
    return compare_kwargs
