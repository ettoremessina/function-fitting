#!/bin/sh
EXM=example2_esvr
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

FX="np.sin(x)"
XB=-6.0
XE=6.0
NS="0.5 * np.random.normal(0.0, 1.0, size=sz)"

python ../../common/fx_gen.py --dsout datasets/${EXM}_train.csv --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.01 --noise "$NS"

python ../../../svr/fit_func_esvr.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 1 \
  --modelout models/${EXM}.jl \

python ../../common/fx_gen.py --dsout datasets/${EXM}_test.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0475

python ../../../svr/predict_func.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 1 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

  python ../../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
  #python ../../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
