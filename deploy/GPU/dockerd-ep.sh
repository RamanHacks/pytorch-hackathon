#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    mkdir model-archive && mkdir model-archive/model_store && mkdir model-archive/wf_store
    torch-model-archiver -f --model-name craft --version 1.0 --serialized-file jit-models/craft_ts.pt --handler det_handler.py --export-path model-archive/model_store/
    torch-model-archiver -f --model-name crnn --version 1.0 --serialized-file jit-models/crnn_ts.pt --handler rec_handler.py --export-path model-archive/model_store/
    cp index_sroie.json index.json
    torch-model-archiver -f --model-name sroie --version 1.0 --serialized-file jit-models/sroie_ts.pt --handler ext_handler.py --export-path model-archive/model_store/ --extra-files index.json
    cp index_funsd.json index.json
    torch-model-archiver -f --model-name funsd --version 1.0 --serialized-file jit-models/funsd_ts.pt --handler ext_handler.py --export-path model-archive/model_store/ --extra-files index.json
    rm index.json
    torch-workflow-archiver -f --workflow-name ocr --spec-file workflow_ocr.yaml --handler workflow_handler.py --export-path model-archive/wf_store/
    torch-workflow-archiver -f --workflow-name ner_sroie --spec-file workflow_ner_sroie.yaml --handler workflow_handler.py --export-path model-archive/wf_store/
    torch-workflow-archiver -f --workflow-name ner_funsd --spec-file workflow_ner_funsd.yaml --handler workflow_handler.py --export-path model-archive/wf_store/
    rm -rf jit-models
    torchserve --start --model-store model-archive/model_store/ --workflow-store model-archive/wf_store/ --ncs --ts-config config.properties
    sleep 2s
    curl -X POST "localhost:7081/workflows?url=ner_funsd.war"
    curl -X POST "localhost:7081/workflows?url=ner_sroie.war"
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null