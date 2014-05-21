#!/bin/sh
#make this chmod u+x , copy this to slaves and run on there
#then start up hdfs on master in persist/bin

mv /vol /vol2
mkdir /vol
mkfs -t ext4 /dev/xvdv
#XFS_MOUNT_OPTS="defaults,noatime,nodiratime,allocsize=8m"
XFS_MOUNT_OPTS=" defaults "
mount -o $XFS_MOUNT_OPTS /dev/xvdv /vol
chmod -R a+w /vol
mv /vol2/* /vol/
rmdir /vol2
echo "/dev/xvdv	/vol	auto	$XFS_MOUNT_OPTS	  0	0" >> /etc/fstab
#not sure if this is idempotent so don't do it
#mount -a
echo "Fixed persistent HDFS issue" > PER_MESSAGE
