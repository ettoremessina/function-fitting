#!/bin/sh
EXM=example1_pycaret
rm -f models/${EXM}.pkl
rm -f predictions/${EXM}_pred.csv

FX="0.5*x**3 - 2*x**2 - 3*x - 1"
XB=-10.0
XE=10.0

python ../../common/fx_gen.py --dsout datasets/${EXM}_train.csv --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.01

python ../../../pycaret/fit_func.py \
  --trainds datasets/${EXM}_train.csv \
  --targetcol y \
  --metric MAE \
  --modelout models/${EXM} \
  --setupparams "'train_size': 0.8, 'session_id': 987654321, 'log_experiment': True, 'experiment_name': '${EXM}'"

python ../../common/fx_gen.py --dsout datasets/${EXM}_test.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0475

python ../../../pycaret/predict_func.py \
 --model models/${EXM} \
 --ds datasets/${EXM}_test.csv \
 --targetcol y \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
#python ../../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
