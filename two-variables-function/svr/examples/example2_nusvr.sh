#!/bin/sh
EXM=example2_nusvr
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

FXY="np.sin(np.sqrt(x**2 + y**2))"
XB=-5.0
XE=5.0
YB=-5.0
YE=5.0
NS="0.5 * np.random.normal(0.0, 1.0, size=sz)"

python ../../common/fxy_gen.py \
  --dsout datasets/${EXM}_train.csv \
  --funcxy "$FXY" \
  --xbegin $XB --xend $XE \
  --ybegin $YB --yend $YE \
  --xstep 0.05 --ystep 0.1 \
  --noise "$NS"

python ../../../svr/fit_func_nusvr.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 1 \
  --modelout models/${EXM}.jl

python ../../common/fxy_gen.py --dsout datasets/${EXM}_test.csv  --funcxy "$FXY" --xbegin $XB --xend $XE --ybegin $YB --yend $YE  --xstep 0.0875 --ystep 0.5

python ../../../svr/predict_func.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 1 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

 python ../../common/fxy_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
 #python ../../common/fxy_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
