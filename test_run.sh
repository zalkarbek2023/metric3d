#!/bin/bash
# ONE pre-registered test run (PREREG.md). A rerun needs its reason written into PREREG.md first.
cd ~/metric3d
P=~/medai-colon/.venv/bin/python
D=~/flatreal/.venv-dp2/bin/python
TEST="0009 0013 0023 0027 0036 0046 0052 0056 0064 0084 0093"
date; sha256sum kitti.py run3d.py score3d.py dp2_kitti.py calib_scale.py worst3d.py test_run.sh data/val6_letterbox.csv \
  ~/flatreal/engines/yolo26x-seg-fp16.engine ~/flatreal/engines/yolo26x-depth-fp16.engine
grep -q FETCHDONE logs_fetch.txt || { echo "download not finished - stop"; exit 1; }
for n in $TEST; do
  d=kitti/2011_09_26/2011_09_26_drive_${n}_sync
  [ -f $d/tracklet_labels.xml ] && [ -d $d/velodyne_points/data ] && [ -d $d/oxts/data ] || { echo "missing $n - stop"; exit 1; }
done
ls data/test_* 2>/dev/null && { echo "test outputs already exist - stop"; exit 1; }
date; $P run3d.py --drives $TEST --variant letterbox --out data/test_letterbox.csv > logs_test_run.txt 2>&1; echo "run rc=$?"
date; $P score3d.py data/test_letterbox.csv > data/test_score.txt 2>&1; echo "score rc=$?"
date; $P score3d.py data/test_letterbox.csv --scale 0.9897920364109557 > data/test_score_cal.txt 2>&1; echo "score cal rc=$?"
date; $D dp2_kitti.py --drives $TEST > logs_test_dp2.txt 2>&1; echo "dp2 rc=$?"
date; $P run3d.py --drives $TEST --depth-dir data/dp2 --out data/test_dp2.csv > logs_test_run_dp2.txt 2>&1; echo "run dp2 rc=$?"
date; $P score3d.py data/test_dp2.csv > data/test_score_dp2.txt 2>&1; echo "score dp2 rc=$?"
date; $P worst3d.py data/test_letterbox.csv out/worst_test.jpg --n 12 > logs_test_worst.txt 2>&1; echo "worst rc=$?"
date; echo ALLDONE
