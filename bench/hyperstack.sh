#!/bin/bash
# Hyperstack VM management for GPU benchmarks.
# Usage:
#   ./bench/hyperstack.sh create [flavor]   — spin up a VM (default: n3-RTX-A6000x1)
#   ./bench/hyperstack.sh list              — list running VMs
#   ./bench/hyperstack.sh ssh [vm_id]       — print SSH command
#   ./bench/hyperstack.sh destroy [vm_id]   — terminate VM
#   ./bench/hyperstack.sh ip [vm_id]        — get IP
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../.env" 2>/dev/null || true
API="https://infrahub-api.nexgencloud.com/v1/core"
AUTH="api_key: ${HYPERSTACK_API_KEY:?Set HYPERSTACK_API_KEY in .env}"

cmd="${1:-help}"

case "$cmd" in
  create)
    FLAVOR="${2:-n3-RTX-A6000x1}"
    echo "Creating VM with flavor $FLAVOR..."
    RESP=$(curl -s -X POST "$API/virtual-machines" \
      -H "$AUTH" -H "Content-Type: application/json" \
      -d "{
        \"name\": \"starkdal-bench-$(date +%s)\",
        \"environment_name\": \"default-CANADA-1\",
        \"image_name\": \"Ubuntu Server 22.04 LTS R535 CUDA 12.2 with Docker\",
        \"flavor_name\": \"$FLAVOR\",
        \"key_name\": \"training-key\",
        \"count\": 1,
        \"assign_floating_ip\": true,
        \"create_bootable_volume\": false
      }")
    echo "$RESP" | python3 -m json.tool
    ;;

  list)
    curl -s -H "$AUTH" "$API/virtual-machines" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for vm in data.get('virtual_machines', []):
    ip = ''
    if vm.get('floating_ip'):
        ip = vm['floating_ip']
    elif vm.get('fixed_ips'):
        ip = vm['fixed_ips'][0].get('ip', '')
    status = vm.get('status', '?')
    print(f'{vm[\"id\"]:8d}  {vm[\"name\"]:40s}  {status:12s}  {ip}  gpu={vm.get(\"flavor\",{}).get(\"gpu\",\"?\")}')
"
    ;;

  ssh)
    VM_ID="${2:?Usage: hyperstack.sh ssh <vm_id>}"
    IP=$(curl -s -H "$AUTH" "$API/virtual-machines/$VM_ID" | python3 -c "
import json, sys
vm = json.load(sys.stdin)['virtual_machine']
print(vm.get('floating_ip') or vm.get('fixed_ips',[{}])[0].get('ip',''))
")
    echo "ssh -o StrictHostKeyChecking=no ubuntu@$IP"
    ;;

  ip)
    VM_ID="${2:?Usage: hyperstack.sh ip <vm_id>}"
    curl -s -H "$AUTH" "$API/virtual-machines/$VM_ID" | python3 -c "
import json, sys
vm = json.load(sys.stdin)['virtual_machine']
print(vm.get('floating_ip') or vm.get('fixed_ips',[{}])[0].get('ip',''))
"
    ;;

  destroy)
    VM_ID="${2:?Usage: hyperstack.sh destroy <vm_id>}"
    echo "Destroying VM $VM_ID..."
    curl -s -X DELETE -H "$AUTH" "$API/virtual-machines/$VM_ID" | python3 -m json.tool
    ;;

  *)
    echo "Usage: ./bench/hyperstack.sh {create|list|ssh|destroy|ip} [args]"
    echo ""
    echo "Flavors (CANADA-1, in stock):"
    echo "  n3-RTX-A6000x1    — RTX A6000 48GB, 28 CPU, 58GB RAM"
    echo "  n3-L40x1          — L40 48GB, 28 CPU, 58GB RAM"
    echo "  n3-A100x1         — A100 80GB, 28 CPU, 120GB RAM"
    echo "  n3-H100x1         — H100 80GB, 28 CPU, 180GB RAM"
    echo "  n3-A100x1-spot    — A100 80GB spot (cheaper)"
    echo "  n3-RTX-A6000x1-spot — A6000 spot (cheapest GPU)"
    ;;
esac
