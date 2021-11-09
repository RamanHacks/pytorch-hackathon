# [pytorch-hackathon] Document Extraction Tool (DET)

DET is an end-to-end tool for extracting **Key-Value** pairs from a variety of documents, built entirely on [PyTorch](https://pytorch.org/) and served using [TorchServe](https://pytorch.org/serve/).

**Try it live here, [web-app]([docs/README.md](https://pytorch-hackathon-4819e.web.app/)).** :wave:

## DET Architecture

The Document Extraction tool is composed of two main components, namely, <br>1. OCR (Optical Character Recognition) and <br>2. Document NER (Named Entity Recognition) <br>
The OCR component comprises of a Detection and Recognition module which work sequentially to produce results which is then consumed by the NER part for training/prediction.

For the OCR part, we have deployed a `torchserve` model server on `GCP` using `Vertex AI` service. Using `torchserve`, we deploy a complex `workflow` in the form of a DAG, comprising of `pre_processing`, `detection` and `recognition` models.

For the NER part, we created a training module using `transformers` library which requires the text and bounding-box results from OCR output to train/predict documents.

The architectural flow of the two modules is shown ![here](https://media.discordapp.net/attachments/900413429381210155/905519648815067196/Result.jpg?width=810&height=678)

### Components:

* **Optical Character Recognition (OCR)**:
  * Detection:
  * Recognition:
* **Named Entity Recognition (NER)**:
  * Receipt Dataset (SROIE):
  * Forms Dataset (FUNSD):


## Contents
- [[pytorch-hackathon]Document Extraction Tool (DET)](#pytorch-hackathondocument-extraction-tool-det)
  - [DET Architecture](#det-architecture)
    - [Components:](#components)
  - [Contents](#contents)
  - [File Structure](#file-structure)
  - [Live Demo](#live-demo)
  - [Deployment](#deployment)
    - [Quick Deploy](#quick-deploy)
    - [Using Docker Containers](#using-docker-containers)
    - [From Source](#from-source)
  - [Sample Request](#sample-request)
    - [Using CURL](#using-curl)
    - [From Python file](#from-python-file)
  - [Training](#training)
    - [Custom NER](#custom-ner)
    - [Custom OCR](#custom-ocr)
  - [Model Optimization](#model-optimization)
  - [Support Our Work](#support-our-work)
  - [License](#license)
  
## File Structure

```
deploy                           # INSTRUCTIONS/SCRIPTS for deploying model(s) on GPU/CPU
|---GPU                          # For Serving models on GPU   
|------jit-models                # contains jitted models created after tracing the model using torch.jit.trace   
|------------craft_ts.pt         # DOWNLOAD the torchscript files from here: https://drive.google.com/drive/folders/1NBSZIZzSzIVOUqnxu0PHgmy-_Tvvp2hY?usp=sharing
|------------crnn_ts.pt    
|------------sroie_ts.pt    
|------------funsd_ts.pt    
|------model-archive             # stores models along with configuration in .mar format as required by "torchserve"  
|------------model_store         # GENERATE standalone .mar files from torch-model-archiver command given below  
|------------------craft.mar     
|------------------crnn.mar
|------------------sroie.mar
|------------------funsd.mar
|------------wf_store            # GENERATE workflow .mar files from torch-workflow-archiver command given below
|------------------ocr.mar
|------------------ner_sroie.mar
|------------------ner_funsd.mar
|------config.properties         # config.properties for storing serving related configurations. 
|                                For details, refer this: https://github.com/pytorch/serve/blob/master/docs/configuration.md
|------detection_handler.py      # handler for text detection pipeline 
|------rec_handler.py            # handler for text recognition pipeline 
|------workfow_handler.py        # handler for configuring end2end ocr serving pipeline 
|------workflow_ocr.yaml         # define the pipeline here, also specify batch-size, num workers etc.
|------workflow_ner_sroie.yaml
|------workflow_ner_funsd.yaml
|------Dockerfile                # Dockerfile template for creating CPU image for UCR
|------cloud_deploy.ipynb        # Scripts to deploy the file on GCP Vertex ai and run predictions.
|------sample_b64.json           # Sample file to send request on inference api

|---CPU                          # Same as above except it's CPU (currently INCOMPLETE)
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

train                            # NOTEBOOKS for training models on GPU/CPU. More training scripts COMING SOON!
|---NER.ipynb                    # Jupyter Notebook to train,test Document NER models and convert it to torchscript format.
```

## Live Demo


## Deployment 

### Quick Deploy  
> Install docker and nvidia container toolkit: See this [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for help!  

> Download and start docker:  
```docker run -d --gpus all -p 7080:7080 -p 7081:7081 -p 7082:7082 abhigyanr/det-gpu:latest```  

In order to send sample request to it, go [here](#sample-request)  
Note: This requires NVIDIA GPU and driver to be present!

### Using Docker Containers  
> Install docker and nvidia container toolkit: See this [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for help!

> Clone this repository and change directory:
```
git clone https://github.com/RamanHacks/pytorch-hackathon.git
cd pytorch-hackathon && cd deploy
cd GPU      # OR (cd CPU)
```

> Build Docker Image
```
docker build -f Dockerfile -t det .
```
> Run Docker container
```
docker run -d --name det-cpu -p 7080:7080 -p 7081:7081 -p 7082:7082 det                # (For CPU)
docker run -d --name det-gpu --gpus all -p 7080:7080 -p 7081:7081 -p 7082:7082 det     # (For GPU, use --gpus '"device=0,1"' to specify device)
```
> Optional: Check Status
```
docker logs $(docker ps -l -q)      # to check if the docker container is running fine
curl localhost:7080/ping            # to check if the network is accessible from localhost, should return Healthy
```

> Register Models
```
curl -X POST "localhost:7081/workflows?url=ocr.war"
curl -X POST "localhost:7081/workflows?url=ner_sroie.war"
curl -X POST "localhost:7081/workflows?url=ner_funsd.war"
```
(Optional: test_run)
```
curl -X POST -H "Content-Type: application/json; charset=utf-8" -d @sample_b64.json localhost:7080/wfpredict/ocr -o output_test.json
```
> Optional: Stop and Remove Container
```
docker stop $(docker ps -l -q)
docker rm $(docker ps -l -q)
```
In order to send sample request to it, go [here](#sample-request)

### From Source  

> Install torch from official link: [PyTorch Official](https://pytorch.org/get-started/locally/)
> Install torchserve from official repo: [TorchServe Official](https://github.com/pytorch/serve.git)
> Install python dependencies:  ```pip install -r requirements.txt```
> Download pretrained torchscript models from [Google Drive](https://drive.google.com/drive/folders/1NBSZIZzSzIVOUqnxu0PHgmy-_Tvvp2hY?usp=sharing)

> Generate .mar files:
```
cd deploy/GPU       #(cd deploy/CPU)
torch-model-archiver -f --model-name craft --version 1.0 --serialized-file jit-models/craft_ts.pt --handler det_handler.py --export-path model-archive/model_store/
torch-model-archiver -f --model-name crnn --version 1.0 --serialized-file jit-models/crnn_ts.pt --handler rec_handler.py --export-path model-archive/model_store/
torch-model-archiver -f --model-name sroie --version 1.0 --serialized-file jit-models/sroie_ts.pt --handler ext_handler.py --export-path model-archive/model_store/
torch-model-archiver -f --model-name funsd --version 1.0 --serialized-file jit-models/funsd_ts.pt --handler ext_handler.py --export-path model-archive/model_store/
torch-workflow-archiver -f --workflow-name ocr --spec-file workflow.yaml --handler workflow_handler.py --export-path model-archive/wf_store/
torch-workflow-archiver -f --workflow-name ner_sroie --spec-file workflow_ner_sroie.yaml --handler workflow_handler.py --export-path model-archive/wf_store/
torch-workflow-archiver -f --workflow-name ner_funsd --spec-file workflow_ner_funsd.yaml --handler workflow_handler.py --export-path model-archive/wf_store/
```
> Start Model Server
```
torchserve --start --model-store model-archive/model_store/ --workflow-store model-archive/wf_store/ --ncs --ts-config config.properties
```
> Register Models
```
curl -X POST "localhost:7081/workflows?url=ocr.war"
curl -X POST "localhost:7081/workflows?url=ner_sroie.war"
curl -X POST "localhost:7081/workflows?url=ner_funsd.war"
```
> Stop TorchServe
```
torchserve --stop
```
In order to send sample request to it, go [here](#sample-request)

## Sample Request  
> Request format: json file containing base64 values of image
```
{
    'data': '<base64 value of an image>' 
}
```

> Response format of OCR(only), i.e. when hitting "/wfpredict/ocr": 
```
[
    {
        'bbox': [[<top-left>],[<top-right>],[<bottom-left>],[<bottom-right>]]
        'ocr': [<value>, <confidence>]
    }
]
```
> Response format of OCR+NER, i.e. when hitting "/wfpredict/ner_sroie" or "/wfpredict/ner_funsd":
```
[
    {
        'bbox': [<top-left-x>,<top-left-y>,<bottom-right-x>,<bottom-right-y>]
        'ocr': <value>
        'key': <value>
    }
]
```

### Using CURL

> Sample CURL Request
```
curl -X POST -H "Content-Type: application/json; charset=utf-8" -d @sample_b64.json localhost:7080/wfpredict/ner_sroie
```

### From Python file

> Python function to convert an image into base64, send request and return predictions
```
def sample_request(image_file_path)
    import base64
    import requests

    def convert_b64(image_file):
        """Open image and convert it to Base64"""
        with open(image_file, "rb") as input_file:
            jpeg_bytes = base64.b64encode(input_file.read()).decode("utf-8")
        return jpeg_bytes

    req = {"data": convert_b64(image_file_path)}
    res = requests.post("http://localhost:7080/wfpredict/ner_sroie", json=req)

    return res.json()
```

## Training

### Custom NER
Jump to [NER.ipynb](https://github.com/RamanHacks/pytorch-hackathon/blob/main/NER/NER.ipynb) for details on training and testing Document NER models!

### Custom OCR
-----Coming-Soon-----
## Model Optimization
-----Coming Soon----- 
## Support Our Work
-----If you like our work, do not forget to :star: this repository and follow us on [twitter](twitter), [linkedin](linkedin)-----
-----If you have got any specific feature request, contact us at admin@docyard.ai----- 
## License
[Apache License 2.0](LICENSE)