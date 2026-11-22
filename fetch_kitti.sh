#!/bin/bash
# Download the 17 KITTI Eigen-test raw drives that have 3D tracklets (excluded from YOLO26-depth training),
# keep only image_02 + velodyne_points + oxts, delete the zips. Same S3 links depth-kitti.yaml uses.
# Val drives first (every third in sorted order), then test. Resumable: curl -C -, skips drives already extracted.
set -u
cd ~/metric3d && mkdir -p kitti zips
base=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
VAL="0002 0020 0029 0048 0059 0086"
TEST="0009 0013 0023 0027 0036 0046 0052 0056 0064 0084 0093"
get() {  # resume after stalls: abort below 50 KB/s for 30 s, then continue from the partial file
  for k in 1 2 3 4 5 6 7 8 9 10; do
    curl -s -f -L -C - --speed-limit 50000 --speed-time 30 -o "zips/$(basename "$1")" "$1" && return 0
    echo "$(date +%H:%M:%S) retry $k $(basename "$1")"; sleep 5
  done
  echo "FAIL $1"; return 1
}
[ -f kitti/2011_09_26/calib_cam_to_cam.txt ] || { get $base/2011_09_26_calib.zip && unzip -qo zips/2011_09_26_calib.zip -d kitti && rm zips/2011_09_26_calib.zip; }
for n in $VAL $TEST; do
  d=2011_09_26_drive_$n
  [ -f kitti/2011_09_26/${d}_sync/tracklet_labels.xml ] || { get $base/$d/${d}_tracklets.zip && unzip -qo zips/${d}_tracklets.zip -d kitti && rm zips/${d}_tracklets.zip; }
  # 0093's tracklets zip has no folders: the xml lands in kitti/ - move it into its drive folder
  [ -f kitti/tracklet_labels.xml ] && mkdir -p kitti/2011_09_26/${d}_sync && mv kitti/tracklet_labels.xml kitti/2011_09_26/${d}_sync/
  if [ ! -d kitti/2011_09_26/${d}_sync/velodyne_points ]; then
    get $base/$d/${d}_sync.zip || continue
    unzip -qo zips/${d}_sync.zip "*/image_02/*" "*/velodyne_points/*" "*/oxts/*" -d kitti && rm zips/${d}_sync.zip
  fi
  echo "$(date +%H:%M:%S) $d done: $(ls kitti/2011_09_26/${d}_sync/image_02/data | wc -l) frames, $(du -sh kitti | cut -f1) total"
done
echo FETCHDONE
