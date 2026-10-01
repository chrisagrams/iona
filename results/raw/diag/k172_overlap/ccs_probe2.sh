#!/bin/bash
cd /home/khuss/code/msdelta; export FRAMEWORKS_MODULE=frameworks/2026.1.0; source pbs/lib/load_frameworks.sh >/dev/null 2>&1
PY=/home/khuss/code/msdelta/.venv-2026/bin/python
probe() { echo "== $*"; env -u ZE_AFFINITY_MASK -u ONEAPI_DEVICE_SELECTOR -u ZEX_NUMBER_OF_CCS -u ZE_FLAT_DEVICE_HIERARCHY "$@" timeout 90 $PY -c 'import torch; n=torch.xpu.device_count(); print("count", n, [(torch.xpu.get_device_properties(i).gpu_eu_count) for i in range(min(n,6))])' 2>&1 | grep -E "^count|Error" | tail -1; }
probe ONEAPI_DEVICE_SELECTOR='level_zero:gpu'
probe ONEAPI_DEVICE_SELECTOR='*:*.*' ZEX_NUMBER_OF_CCS=0:2 ZE_FLAT_DEVICE_HIERARCHY=FLAT
probe ONEAPI_DEVICE_SELECTOR='level_zero:0.*' ZEX_NUMBER_OF_CCS=0:2 ZE_FLAT_DEVICE_HIERARCHY=FLAT
probe ONEAPI_DEVICE_SELECTOR='*:*.*.*' ZEX_NUMBER_OF_CCS=0:2 ZE_FLAT_DEVICE_HIERARCHY=FLAT
probe ONEAPI_DEVICE_SELECTOR='*:*.*' ZEX_NUMBER_OF_CCS=0:4 ZE_FLAT_DEVICE_HIERARCHY=FLAT
probe ONEAPI_DEVICE_SELECTOR='*:0.*' ZEX_NUMBER_OF_CCS=0:2
probe ZEX_NUMBER_OF_CCS=0:2 ZE_FLAT_DEVICE_HIERARCHY=FLAT
probe ONEAPI_DEVICE_SELECTOR='*:*.*.*' ZEX_NUMBER_OF_CCS=0:2 ZE_FLAT_DEVICE_HIERARCHY=COMBINED
echo DONE
