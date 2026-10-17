#!/usr/bin/env bash
# Fetch source scores. Not redistributed here: the WTC encodings are
# Copyright 1994 David Huron, derivative electronic formats reserved.
set -euo pipefail
cd "$(dirname "$0")/../data"
[ -d bach-wtc ] || git clone --depth 1 https://github.com/humdrum-tools/bach-wtc
[ -d art-of-the-fugue ] || git clone --depth 1 https://github.com/craigsapp/art-of-the-fugue
