#!/bin/sh
EXM=example1
rm -f models/${EXM}.pkl
rm -f predictions/${EXM}_pred.csv

FXY="x**2 + y**2"
XB=-3.0
XE=3.0
YB=-3.0
YE=3.0

python ../../common/fxy_gen.py \
  --dsout datasets/${EXM}_train.csv \
  --funcxy "$FXY" \
  --xbegin $XB --xend $XE \
  --ybegin $YB --yend $YE \
  --xstep 0.05 --ystep 0.1

python ../../../pycaret/fit_func.py \
  --trainds datasets/${EXM}_train.csv \
  --targetcol z \
  --modelout models/${EXM}.pkl \
  --setupparams "'train_size': 0.8, 'session_id': 987654321, 'log_experiment': True, 'experiment_name': '${EXM}'"

python ../../common/fxy_gen.py \
  --dsout datasets/${EXM}_test.csv \
  --funcxy "$FXY" \
  --xbegin $XB --xend $XE \
  --ybegin $YB --yend $YE  \
  --xstep 0.0875 --ystep 0.5

python ../../../pycaret/predict_func.py \
  --model models/${EXM}.pkl \
  --ds datasets/${EXM}_test.csv \
  --targetcol z \
  --measures mean_absolute_error mean_squared_error \
  --predictionout predictions/${EXM}_pred.csv

 python ../../common/fxy_scatter.py \
   --ds datasets/${EXM}_test.csv \
  --prediction predictions/${EXM}_pred.csv \
  --xlabel "x" \
  --ylabel "y" \
  --zlabel "z=$FXY"
