#!/bin/sh
SLAVES=`cat /root/spark-ec2/slaves`
FIX_SCRIPT=./fix-persist-spark-ec2.sh
for slave in $SLAVES; do
    echo "fixing slave $slave"
    chmod u+x $FIX_SCRIPT
    scp $FIX_SCRIPT $slave:
    ssh $slave $FIX_SCRIPT
    ssh $slave rm $FIX_SCRIPT
done
