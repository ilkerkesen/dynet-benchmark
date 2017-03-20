#!/bin/bash

export ANACONDA_PATH=$HOME/usr/local/anaconda3/envs/benchmark2
export CUDA_PATH=/usr/local/cuda
export DYNET_PATH=$HOME/work/dynet
export LD_LIBRARY_PATH=$DYNET_PATH/build/dynet:$ANACONDA_PATH/lib:$CUDA_PATH/lib64
export LIBRARY_PATH=$DYNET_PATH/build/dynet:$ANACONDA_PATH/lib:$CUDA_PATH/lib64
export PYTHONPATH=$DYNET_PATH/build/python
PYTHON=python
JULIA=julia

DYFLAGS="--dynet-mem 4096"
GPUSUF=
if [[ $# == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=$1
  export THEANO_FLAGS="device=gpu0,floatX=float32"
  DYFLAGS="$DYFLAGS --dynet-gpus 1"
  GPUSUF="-gpu"
  CGPU=0
  KNETGPU=" --gpu"
else
  export THEANO_FLAGS="device=cpu,floatX=float32"
  CGPU=-1
  KNETGPU=""
fi

TIMEOUT=600
LONGTIMEOUT=600

runcmd() {
  LFILE=log/$2$GPUSUF/$4.log
  if [[ ! -e $LFILE ]]; then
    MYTIMEOUT=$TIMEOUT
    if [[ $1 == "dynet-cpp" ]]; then
      mycmd="$1/$2$GPUSUF $DYFLAGS"
      if [[ $4 =~ dynet-cpp-bs01-ws128-hs256-.* ]] || [[ $4 =~ dynet-cpp-bs16-ws128-hs256-.* ]] || [[ $2 =~ bilstm.* ]] || [[ $2 =~ treenn ]]; then
        MYTIMEOUT=$LONGTIMEOUT
      fi
    elif [[ $1 == "dynet-py" ]]; then
      mycmd="$PYTHON -u $1/$2.py $DYFLAGS"
    elif [[ $1 == "chainer" ]]; then
      mycmd="$PYTHON -u $1/$2.py --chainer_gpu $CGPU"
    elif [[ $1 == "tensorflow" ]]; then
	mycmd="$PYTHON -u $1/$2.py --gpu"
    elif [[ $1 == "knet" ]]; then
	mycmd="$JULIA $1/$2.jl"$KNETGPU
    else
      mycmd="$PYTHON -u $1/$2.py"
    fi
    mkdir -p log/$2$GPUSUF
    echo "$mycmd $3 $MYTIMEOUT &> $LFILE"
    eval "$mycmd $3 $MYTIMEOUT &> $LFILE"
  fi
}

for trial in 1 2 3; do

  # Run rnnlm-batch
  for embsize in 128; do
    hidsize=$(($embsize*2))
    for mbsize in 64 16 04 01; do
      for f in dynet-cpp dynet-py chainer theano tensorflow knet; do
        if [[ $f == "dynet-cpp" ]]; then
          runcmd $f rnnlm-seq "$mbsize $embsize $hidsize 0" $f-ms$mbsize-es$embsize-hs$hidsize-sp0-t$trial
        fi
        runcmd $f rnnlm-batch "$mbsize $embsize $hidsize 0" $f-ms$mbsize-es$embsize-hs$hidsize-sp0-t$trial
      done
    done
  done

  # run sparse rnnlm-batch on a subset
  for embsize in 128; do
    hidsize=$(($embsize*2))
    for mbsize in 16 01; do
      for f in dynet-cpp dynet-py; do
        runcmd $f rnnlm-batch "$mbsize $embsize $hidsize 1" $f-ms$mbsize-es$embsize-hs$hidsize-sp1-t$trial
      done
    done
  done

  # Run bilstm-tagger
  wembsize=128
  hidsize=50
  mlpsize=32
  for f in dynet-cpp dynet-py chainer theano tensorflow knet; do
    runcmd $f bilstm-tagger "$wembsize $hidsize $mlpsize 0" $f-ws$wembsize-hs$hidsize-mlps$mlpsize-su0-t$trial
    if [[ $f == dynet* ]]; then
      runcmd $f bilstm-tagger "$wembsize $hidsize $mlpsize 1" $f-ws$wembsize-hs$hidsize-mlps$mlpsize-su1-t$trial
    fi
  done

  # Run bilstm-tagger-withchar
  cembsize=20
  wembsize=128
  hidsize=50
  mlpsize=32
  for f in dynet-cpp dynet-py theano chainer knet; do
    runcmd $f bilstm-tagger-withchar "$cembsize $wembsize $hidsize $mlpsize 0" $f-cs$cembsize-ws$wembsize-hs$hidsize-mlps$mlpsize-su0-t$trial
    if [[ $f == dynet* ]]; then
      runcmd $f bilstm-tagger-withchar "$cembsize $wembsize $hidsize $mlpsize 1" $f-cs$cembsize-ws$wembsize-hs$hidsize-mlps$mlpsize-su1-t$trial
    fi
  done

  # Run treenn
  wembsize=128
  hidsize=128
  for f in dynet-cpp dynet-py chainer knet; do
    runcmd $f treenn "$wembsize $hidsize 0" $f-ws$wembsize-hs$hidsize-su0-t$trial
    if [[ $f == dynet* ]]; then
      runcmd $f treenn "$wembsize $hidsize 1" $f-ws$wembsize-hs$hidsize-su1-t$trial
    fi
  done

done
