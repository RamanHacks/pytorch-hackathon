# [pytorch-hackathon] Document Extraction Tool
The Document Extraction tool is composed of two separate components, namely, <br>1. OCR (Optical Character Recognition) and <br>2. Document NER (Named Entity Recognition) <br>

The OCR component comprises of a Detection and Recognition module which work sequentially to produce results which is then consumed by the NER part for training/prediction.

For the OCR part, we have deployed a `torchserve` model server on `GCP` using `Vertex AI` service. Using `torchserve`, we deploy a complex `workflow` in the form of a DAG, comprising of `pre_processing`, `detection` and `recognition` models.

For the NER part, we created a training module using `transformers` library which requires the text and bounding-box results from OCR output to train/predict documents.

The architectural flow of the two modules is shown here:

# File Structure

```
OCR
|---CPU                          # For Serving models on CPU   
|------jit-models                # jitted models created after tracing the model using torch.jit.trace   
|------------craft_ts.pt   
|------------crnn_ts.pt   
|------model-archive             # model-archive stores models along with configuration in .mar format as required by "torchserve"  
|------------model_store         # it stores standalone .mar files created from torch-model-archiver  
|------------------craft.mar     
|------------------crnn.mar
|------------wf_store            # it contains workflow .mar file created from torch-workflow-archiver
|------------------ocr.mar
|------config.properties         # config.properties for storing serving related configurations. 
|                                For details, refer this: https://github.com/pytorch/serve/blob/master/docs/configuration.md
|------detection_handler.py      # handler for text detection pipeline 
|------rec_handler.py            # handler for text recognition pipeline 
|------workfow_handler.py        # handler for configuring end2end ocr serving pipeline 
|------workflow.yaml             # define the pipeline here, also specify batch-size, num workers etc.
|------Dockerfile                # Dockerfile to create CPU image for UCR
|------deploy.ipynb              # Scripts to deploy the file on GCP Vertex ai and run predictions.
|---GPU                          # Same as above except it's GPU
|------jit-models
|------------craft_ts.pt
|------------crnn_ts.pt
|------model-archive
|------------model_store         
|------------------craft.mar
|------------------crnn.mar
|------------wf_store            
|------------------ocr.mar
|------config.properties
|                                
|------detection_handler.py
|------rec_handler.py
|------workfow_handler.py
|------workflow.yaml
|------Dockerfile      

NER
|---NER.ipynb                    # Jupyter Notebook to train and test Document NER models.
```

# OCR Part: Commands to Run

## Standalone (with't Docker)

### Assuming that you have already created python env and installed required packages (torchserve, pytorch, ucr, etc.), follow the steps

cd CPU (or GPU)

(Optional: if you want to rebuild the .mar packages)
```
torch-model-archiver -f --model-name craft --version 1.0 --serialized-file jit-models/craft_ts.pt --handler det_handler.py --export-path model-archive/model_store/

torch-model-archiver -f --model-name crnn --version 1.0 --serialized-file jit-models/crnn_ts.pt --handler rec_handler.py --export-path model-archive/model_store/

torch-workflow-archiver -f --workflow-name ocr --spec-file workflow.yaml --handler workflow_handler.py --export-path model-archive/wf_store/
```
(Start Model Server)
```
torchserve --start --model-store model-archive/model_store/ --workflow-store model-archive/wf_store/ --ncs
```
(Register Models)
```
curl -X POST "localhost:7081/workflows?url=ocr.war"
```
(Optional: test_run)
```
curl -X POST -H "Content-Type: application/json; charset=utf-8" -d @sample_b64.json localhost:7080/wfpredict/ocr -o output_test.json
```
(Stop torchserve)
torchserve --stop

## Using Docker serve

### First, install Docker and nvidia-docker2 (for gpu serving) and then follow the steps below

cd CPU (or GPU)

(Build Docker Image)
```
docker build -f Dockerfile -t ocr ./
```
(Run Docker container)
```
docker run -d -p 7080:7080 -p 7081:7081 -p 7082:7082 ucr                # (For CPU)
docker run -d --gpus all -p 7080:7080 -p 7081:7081 -p 7082:7082 ucr     # (For GPU, use --gpus all or --gpus '"device=0,1"')
```
(Optional: check status)
```
docker logs $(docker ps -l -q)      # Check if the docker container is running fine
curl localhost:7080/ping            # Check the network status, should return Healthy
```

(Register Models)
```
curl -X POST "localhost:7081/workflows?url=ocr.war"
```
(Optional: test_run)
```
curl -X POST -H "Content-Type: application/json; charset=utf-8" -d @sample_b64.json localhost:7080/wfpredict/ocr -o output_test.json
```
(Optional: stop and remove container)
```
docker stop $(docker ps -l -q)
docker rm $(docker ps -l -q)
```

# NER Part

Jump to [NER.ipynb]() for details on training and testing Document NER models!