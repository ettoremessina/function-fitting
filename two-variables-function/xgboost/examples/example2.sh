#!/bin/sh
EXM=example2
rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -f predictions/${EXM}_pred.csv

FXY="np.sin(np.sqrt(x**2 + y**2))"
XB=-5.0
XE=5.0
YB=-5.0
YE=5.0

python ../../common/fxy_gen.py --dsout datasets/${EXM}_train.csv --funcxy "$FXY" --xbegin $XB --xend $XE --ybegin $YB --yend $YE --xstep 0.1 --ystep 0.1
python ../../../xgboost/fit_func_miso.py \
  --trainds datasets/${EXM}_train.csv \
  --modelout models/${EXM}.jl

python ../../common/fxy_gen.py --dsout datasets/${EXM}_test.csv  --funcxy "$FXY" --xbegin $XB --xend $XE --ybegin $YB --yend $YE  --xstep 0.475 --ystep 0.475
python ../../../xgboost/predict_func_miso.py \
  --model models/${EXM}.jl \
  --ds datasets/${EXM}_test.csv \
  --predictionout predictions/${EXM}_pred.csv

python ../../common/fxy_scatter.py \
  --ds datasets/${EXM}_test.csv \
  --prediction predictions/${EXM}_pred.csv \
  --title "XGBoost (all properties defaulted)" \
  --xlabel "x" \
  --ylabel "y" \
  --zlabel "z=sin(sqrt(x^2 + y^2))"
#python ../fxy_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png

#python ../fxy_video.py --modelsnap snaps/${EXM} --ds datasets/${EXM}_test.csv --savevideo media/${EXM}_test.gif
