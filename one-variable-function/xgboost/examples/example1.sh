#!/bin/sh
EXM=example1
rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -f predictions/${EXM}_pred.csv

FX="0.5*x**3 - 2*x**2 - 3*x - 1"
XB=-10.0
XE=10.0

python ../../common/fx_gen.py --dsout datasets/${EXM}_train.csv --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.01
python ../../common/fx_gen.py --dsout datasets/${EXM}_val.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0875

python ../../../xgboost/fit_func_miso.py \
  --trainds datasets/${EXM}_train.csv \
  --modelout models/${EXM}.jl \
  --valds datasets/${EXM}_val.csv \
  --xgbparams "'n_estimators':20, 'max_depth':5, 'booster':'dart'"

python ../../common/fx_gen.py --dsout datasets/${EXM}_test.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0475

python ../../../xgboost/predict_func_miso.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
#python ../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png

#python ../fx_video.py --modelsnap snaps/${EXM} --ds datasets/${EXM}_test.csv --savevideo media/${EXM}_test.gif
