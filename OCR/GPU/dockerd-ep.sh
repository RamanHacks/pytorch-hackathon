#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --model-store=/home/model-server/model-store/model_store --workflow-store=/home/model-server/model-store/wf_store --ts-config /home/model-server/config.properties
    sleep 10s
    curl -X POST "localhost:7081/workflows?url=ocr.war"
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null