#!/bin/bash
# Arm test run 2 (PREREG "Повторный прогон test"): wait for run 1's chain, archive it in run1/, install the fixed code, check hashes, run once.
cd ~/metric3d
L=logs_run2_arm.txt
until grep -q ALLDONE logs_test.txt; do sleep 20; done
echo "$(date +%T) run 1 chain finished" >> $L
mkdir -p run1/data run1/out
mv logs_test.txt logs_test_run.txt logs_test_dp2.txt logs_test_run_dp2.txt logs_test_worst.txt logs_test_precheck_stop_1859.txt run1/ 2>>$L
mv data/test_* run1/data/ 2>>$L
[ -f out/worst_test.jpg ] && mv out/worst_test.jpg run1/out/
cp ~/metric3d_fix/kitti.py ~/metric3d_fix/test_run.sh .
printf '%s  kitti.py\n%s  test_run.sh\n' 40df3656a2eaac89d866a66dfa538092150374ad2fca0637ffa8c3130b62dc82 02b6c270336efb801733fb765cdef3174f58f972eda19d2a474a9382bd27f90a | sha256sum -c >> $L 2>&1 || { echo "hash mismatch - stop" >> $L; exit 1; }
echo "$(date +%T) run 2 start" >> $L
./test_run.sh > logs_test.txt 2>&1
echo "$(date +%T) run 2 exit $?" >> $L
