#!/bin/sh

ulimit -c 0

cd ${HOME}/DQM/dqm-GUI

LD_LIBRARY_PATH=
source rpms/cmsset_default.sh
source rpms/slc4_ia32_gcc345/cms/dqmgui/5.1.3/etc/profile.d/env.sh

if [ -e /tmp/createRunIndex.lock ]; then
  echo "Lock file is present, exit"
  exit 1
fi

touch /tmp/createRunIndex.lock

rm -fr /data/ecalod-disk01/dqm-GUI/idx

echo "Index create: begin"

visDQMIndex create /data/ecalod-disk01/dqm-GUI/idx

echo "Index create: end"

rm /tmp/createRunIndex.lock

exit 0

