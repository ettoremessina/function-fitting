#!/bin/sh
EXM=example11
rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -f predictions/${EXM}_pred.csv

FX="np.abs(np.sin(x**x)/(2**(x**x - np.pi/2)/np.pi))"
XB=0.0
XE=3.0

python ../../common/fx_gen.py --dsout datasets/${EXM}_train.csv --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0031
python ../../common/fx_gen.py --dsout datasets/${EXM}_val.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0073

python ../../../xgboost/fit_func_miso.py \
 --trainds datasets/${EXM}_train.csv \
 --modelout models/${EXM}.jl \
 --valds datasets/${EXM}_val.csv

python ../../common/fx_gen.py --dsout datasets/${EXM}_test.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0073

python ../../../xgboost/predict_func_miso.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --predictionout predictions/${EXM}_pred.csv \
  --measures mean_absolute_error mean_squared_error

python ../../common/fx_scatter.py \
  --ds datasets/${EXM}_test.csv \
  --prediction predictions/${EXM}_pred.csv \
  --title "XGBoost (all properties defaulted)" \
  --xlabel "x" \
  --ylabel "y=|sin(x^x)/2^(x^x-pi/2)/pi|"

#python ../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png

#python ../fx_video.py --modelsnap snaps/${EXM} --ds datasets/${EXM}_test.csv --savevideo media/${EXM}_test.gif
