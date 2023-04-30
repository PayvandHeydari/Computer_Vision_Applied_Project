#!/bin/bash
# sudo ./iostat_gpu_to_graphite.sh

GRAPHITE_SERVER=graphite
GRAPHITE_PORT=2003
HOSTNAME=$(hostname)

while true; do
  IOSTAT_OUTPUT=$(iostat -c -x -d -m 1 2 | tail -n +4)
  TIMESTAMP=$(date +%s)
  
  # Collect GPU usage information
  GPU_FREQUENCY=$(timeout 2 sudo intel_gpu_frequency | grep -Eo '[0-9]{1,4} MHz' | awk '{print $1}')


  echo "GPU_FREQUENCY: $GPU_FREQUENCY"

  echo "$HOSTNAME.gpu.gpu_frequency $GPU_FREQUENCY $TIMESTAMP" | nc -q 1 $GRAPHITE_SERVER $GRAPHITE_PORT


  echo "$IOSTAT_OUTPUT" | while read -r LINE; do
    DEVICE=$(echo "$LINE" | awk '{print $1}')
    if [[ $DEVICE != "Device" ]]; then
      FIELDS=($(echo "$LINE"))
      if [[ ${#FIELDS[@]} -ge 14 ]]; then
        UTIL=${FIELDS[13]}
        TPS=${FIELDS[11]}
        MB_READ=${FIELDS[4]}
        RPS=${FIELDS[2]}
        WMB=${FIELDS[6]}

        DEVICE_UNDERSCORED=${DEVICE//./_}

        echo "UTIL: $UTIL"
        echo "TPS: $TPS"
        echo "MB_READ: $MB_READ"
        echo "RPS: $RPS"
        echo "WMB: $WMB"

        echo "$HOSTNAME.disk_${DEVICE_UNDERSCORED}.util $UTIL $TIMESTAMP" | nc -q 1 $GRAPHITE_SERVER $GRAPHITE_PORT
        echo "$HOSTNAME.disk_${DEVICE_UNDERSCORED}.tps $TPS $TIMESTAMP" | nc -q 1 $GRAPHITE_SERVER $GRAPHITE_PORT
        echo "$HOSTNAME.disk_${DEVICE_UNDERSCORED}.mb_read $MB_READ $TIMESTAMP" | nc -q 1 $GRAPHITE_SERVER $GRAPHITE_PORT
        echo "$HOSTNAME.disk_${DEVICE_UNDERSCORED}.rps $RPS $TIMESTAMP" | nc -q 1 $GRAPHITE_SERVER $GRAPHITE_PORT
        echo "$HOSTNAME.disk_${DEVICE_UNDERSCORED}.wmb $WMB $TIMESTAMP" | nc -q 1 $GRAPHITE_SERVER $GRAPHITE_PORT
      fi
    fi
  done

  sleep 60
done
