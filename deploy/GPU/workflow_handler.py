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
        input_data = (row.get("data") or row.get("body") or row)["instances"][0]
        b64_data.append(input_data['data']['b64'])
    return b64_data
