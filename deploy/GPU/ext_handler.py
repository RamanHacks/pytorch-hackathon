from ts.torch_handler.base_handler import BaseHandler
import torch
import os
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
from transformers import LayoutLMTokenizer
import json

class RecHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        # load the model
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)
        
        # preprocess and postprocess initialization
        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.max_seq_length = 512
        model_dir = properties.get("model_dir")
        mapping_file_path = os.path.join(model_dir,"index.json")
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')
        # label_list = ['answer', 'header', 'question', 'answer', 'header', 'question', 'answer', 'header', 'question', '#other', 'answer', 'header', 'question']
        # label_list = ['B-ANSWER', 'B-HEADER', 'B-QUESTION', 'E-ANSWER', 'E-HEADER', 'E-QUESTION', 'I-ANSWER', 'I-HEADER', 'I-QUESTION', '#other', 'S-ANSWER', 'S-HEADER', 'S-QUESTION']
        # self.label_map= dict(zip(range(len(label_list)),label_list))
        
        self.initialized = True
 
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        input_ids = []
        input_mask = []
        segment_ids = []
        token_boxes = []
        ocr_boxes = []
        for row in data:
            val = row.get("recognition") or row.get("body")
            predictions = eval(val.decode())
            df = pd.DataFrame(predictions)
            bboxes = df['bbox'].tolist()
            boxes = np.array(bboxes).astype('float32')
            boxes = [[int(1000 * box[0][0]),int(1000 * box[0][1]),int(1000 * box[2][0]),int(1000 * box[2][1])] for box in bboxes]         
            # boxes = []
            # for box in bboxes:
            #     box = np.array(box).astype('float32')
            #     boxes.append([int(1000 * box[0][0]),int(1000 * box[0][1]),int(1000 * box[2][0]),int(1000 * box[2][1]),])
            words = [w[0] for w in df['ocr'].tolist()]   
            features = self.convert_example_to_features(words=words, boxes=boxes)
            input_ids.append(torch.tensor(features[0]))
            input_mask.append(torch.tensor(features[1]))
            segment_ids.append(torch.tensor(features[2]))
            token_boxes.append(torch.tensor(features[3]))
            ocr_boxes.append(features[4])
        return torch.stack(input_ids).to(self.device), torch.stack(input_mask).to(self.device), torch.stack(segment_ids).to(self.device), torch.stack(token_boxes).to(self.device), ocr_boxes
    
    def convert_example_to_features(self, words, boxes, cls_token_box=[0, 0, 0, 0],
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0]):
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length 
        tokens = []
        token_boxes = []
        ocr_boxes = []
        for word, box in zip(words, boxes):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            ocr_boxes.extend([word]* len(word_tokens))

        # Truncation: account for [CLS] and [SEP] with "- 2". 
        special_tokens_count = 2 
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            ocr_boxes = ocr_boxes[:(max_seq_length - special_tokens_count)]

        # add [SEP] token, with corresponding token boxes and actual boxes
        tokens += [tokenizer.sep_token]
        token_boxes += [sep_token_box]
        ocr_boxes += ['']
        
        segment_ids = [0] * len(tokens)

        # next: [CLS] token
        tokens = [tokenizer.cls_token] + tokens
        token_boxes = [cls_token_box] + token_boxes
        ocr_boxes = [''] + ocr_boxes
        segment_ids = [1] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [tokenizer.pad_token_id] * padding_length
        token_boxes += [pad_token_box] * padding_length
        ocr_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        assert len(ocr_boxes) == max_seq_length
        
        return input_ids, input_mask, segment_ids, token_boxes, ocr_boxes        
    
    def postprocess(self, preds, input_ids, ocrs, bboxes):  
           
        tokenizer = self.tokenizer
        res_list=[]
        # print("Predictions:",preds)
        # print("Input_ids:", input_ids)
        # print("Here"*100,len(preds),len(input_ids))        
        for i in range(len(input_ids)):
            predictions = []
            # print("Here"*10,len(input_ids[i].squeeze().tolist()))
            res = preds[0][i].argmax(-1).squeeze().tolist()
            # print("Here"*100,len(res),len(input_ids[i].squeeze().tolist()))
            last_ocr = ''
            for id, token_pred, tb, ocr in zip(input_ids[i].squeeze().tolist(), res, bboxes[i].squeeze().tolist(), ocrs[i]):
                if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, 
                                                                        tokenizer.sep_token_id, 
                                                                        tokenizer.pad_token_id]):
                    
                    continue
                else:
                    if ocr != last_ocr:
                        last_ocr= ocr
                    else:
                        continue
                    json_result= {}
                    if self.mapping:
                        token_pred = self.mapping[str(token_pred)].split('-')[-1]
                    predicted_label = token_pred
                    rel_box = [b/1000 for b in tb]
                    json_result["bbox"] = rel_box
                    json_result["ocr"] = ocr
                    json_result["key"] = predicted_label   
                    predictions.append(json_result)
                    if predicted_label!='#other':
                        print(json_result)
            res_list.append(predictions)
        return res_list
    
    
    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """
        self.context = context

        input_ids, attention_mask, token_type_ids, bboxes, ocrs = self.preprocess(data)
        model_output = self.inference(input_ids, bboxes, attention_mask, token_type_ids)
        output =  self.postprocess(model_output, input_ids, ocrs, bboxes)
        return output