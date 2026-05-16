#!/bin/bash
# Create a tarball of the repo for deployment to remote machines.
# Excludes build artifacts, git history, and large directories.
# Output: /tmp/starkdal.tar.gz (~2MB)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

tar czf /tmp/starkdal.tar.gz \
    --exclude='target' \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='stwo' \
    --exclude='.env' \
    .

echo "Created /tmp/starkdal.tar.gz ($(du -h /tmp/starkdal.tar.gz | cut -f1))"
