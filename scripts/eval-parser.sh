#!/bin/bash

DDIR=$1
WDIR=$2
STARTI=$3
ENDI=$4
STEPI=$5

i=$STARTI
while [[ $i -le $ENDI ]]; do
  NI="$i"
  echo "DEV "$NI
  $HOME/ndpdp/tools/eval.pl -q -g $DDIR-working/dev.conll -s $WDIR/dev.$NI".conll"
  ((i = i + $STEPI))
done

