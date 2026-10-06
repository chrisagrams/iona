#!/bin/bash
# Install a portable Telegraf binary for Aurora compute nodes.
# Usage: pbs/install-telegraf.sh PREFIX  (e.g. /flare/PROJECT/$USER/telegraf)
# Then pass TELEGRAF_BIN=$PREFIX/bin/telegraf to aurora-pretrain.pbs via qsub -v.
set -euo pipefail

TELEGRAF_VERSION=${TELEGRAF_VERSION:-1.40.1}
PREFIX=${1:-${PREFIX:?pass an install prefix on a shared filesystem}}
ALCF_PROXY=${ALCF_PROXY:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-$ALCF_PROXY}

archive=telegraf-${TELEGRAF_VERSION}_linux_amd64.tar.gz
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

curl --fail --location --silent --show-error \
    -o "$workdir/$archive" "https://dl.influxdata.com/telegraf/releases/$archive"
tar -xzf "$workdir/$archive" -C "$workdir"
mkdir -p "$PREFIX/bin"
install -m 0755 "$workdir"/telegraf-*/usr/bin/telegraf "$PREFIX/bin/telegraf"
"$PREFIX/bin/telegraf" --version
echo "TELEGRAF_BIN=$PREFIX/bin/telegraf"
