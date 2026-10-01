#!/bin/bash
# K177-P: which env combination exposes CCSs as separate PyTorch XPU devices? Run on a compute node.
cd /home/khuss/code/msdelta
export FRAMEWORKS_MODULE=frameworks/2026.1.0; source pbs/lib/load_frameworks.sh >/dev/null 2>&1
PY=/home/khuss/code/msdelta/.venv-2026/bin/python
probe() {
  echo "== $*"
  env -u ZE_AFFINITY_MASK -u ONEAPI_DEVICE_SELECTOR -u ZEX_NUMBER_OF_CCS -u ZE_FLAT_DEVICE_HIERARCHY "$@" \
    timeout 120 $PY -c 'import torch; n=torch.xpu.device_count(); print("count", n, [(torch.xpu.get_device_properties(i).name[-4:], getattr(torch.xpu.get_device_properties(i),"gpu_eu_count",None), round(torch.xpu.get_device_properties(i).total_memory/2**30,1)) for i in range(min(n,6))])' 2>&1 | grep -E "count|Error" | tail -1
}
probe true
probe ONEAPI_DEVICE_SELECTOR='*:*.*.*'
probe ONEAPI_DEVICE_SELECTOR='*:*.*.*' ZEX_NUMBER_OF_CCS=0:2
probe ONEAPI_DEVICE_SELECTOR='*:*.*.*' ZEX_NUMBER_OF_CCS=0:2,1:2
probe ONEAPI_DEVICE_SELECTOR='*:*.*.*' ZEX_NUMBER_OF_CCS=0:4,1:4
probe ONEAPI_DEVICE_SELECTOR='level_zero:*.*.*' ZEX_NUMBER_OF_CCS=0:2
probe ONEAPI_DEVICE_SELECTOR='level_zero:0.*.*' ZEX_NUMBER_OF_CCS=0:2
probe ONEAPI_DEVICE_SELECTOR='*:*.*.*' ZEX_NUMBER_OF_CCS=0:2 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
probe ONEAPI_DEVICE_SELECTOR='*:*.*' ZEX_NUMBER_OF_CCS=0:2 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
