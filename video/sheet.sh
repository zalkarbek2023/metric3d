#!/bin/sh
# sheet.sh out.jpg cols files... : contact sheet via the PC's ffmpeg (the Mac has no image tooling)
out=$1; cols=$2; shift 2
ssh pc 'rm -rf /tmp/sheet && mkdir -p /tmp/sheet'
i=0; for f in "$@"; do scp -q "$f" "pc:/tmp/sheet/$(printf %02d $i).png"; i=$((i+1)); done
rows=$(( (i + cols - 1) / cols ))
ssh pc "ffmpeg -v error -framerate 1 -i /tmp/sheet/%02d.png -vf scale=360:-1,tile=${cols}x${rows} -frames:v 1 -y /tmp/sheet/sheet.jpg" && scp -q pc:/tmp/sheet/sheet.jpg "$out"
