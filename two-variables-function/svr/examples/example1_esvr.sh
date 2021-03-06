#!/bin/sh
EXM=example1_esvr
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

FXY="x**2 + y**2"
XB=-3.0
XE=3.0
YB=-3.0
YE=3.0

python ../../common/fxy_gen.py --dsout datasets/${EXM}_train.csv --funcxy "$FXY" --xbegin $XB --xend $XE --ybegin $YB --yend $YE --xstep 0.05 --ystep 0.1

python ../../../svr/fit_func_esvr.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 1 \
  --modelout models/${EXM}.jl \
  --svrparams "'kernel': 'rbf', 'C': 100, 'gamma': 0.1, 'epsilon': 0.1"

python ../../common/fxy_gen.py --dsout datasets/${EXM}_test.csv  --funcxy "$FXY" --xbegin $XB --xend $XE --ybegin $YB --yend $YE  --xstep 0.0875 --ystep 0.5

python ../../../svr/predict_func.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 1 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

 python ../../common/fxy_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
 #python ../../common/fxy_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
