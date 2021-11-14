import json
import base64
import logging

logger = logging.getLogger(__name__)

def pre_processing(data, context):
    '''
    Empty node as a starting node since the DAG doesn't support multiple start nodes
    '''
    if data is None:
        return data
    b64_data = []
    for row in data:
        input_data = row.get("data") or row.get("body") or row
        b64_data.append(input_data['b64'])
    return b64_data

def post_processing(data, context):
    post_process = []  
    print(data)  
    if data is None:
        return data
    for row in data:
        out_data = row.get("data") or row.get("body") or row.get("extraction")
        post_process.append(out_data.decode())
    return post_process