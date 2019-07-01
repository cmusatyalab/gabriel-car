#! /bin/bash

set -e

GABRIELPATH="/home/jwu96/Desktop/gabriel"

function die { echo $1; exit 42; }
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "working directory $DIR"
# Dependency checks
# source torch if user indicates it's not activated by default
if [ -z ${GABRIELPATH+x} ]
then
   die "Gabriel Not Found. Please specify environment variable GABRIELPATH to be Gabriel's root directory";
else
   echo "User specified Gabriel at ${GABRIELPATH}";
fi

echo "launching Gabriel at ${GABRIELPATH}"
cd $GABRIELPATH/server/bin
./gabriel-control -l -d -n eno1 &> /tmp/gabriel-control.log &
sleep 5
# NOTE: if the default interface is not eth0 network/util.py change get_ip default interface
./gabriel-ucomm -s 0.0.0.0:8021 &> /tmp/gabriel-ucomm.log &
sleep 5

cd ${DIR}
twistd -n web --path resources/videos --port tcp:9095

wait
