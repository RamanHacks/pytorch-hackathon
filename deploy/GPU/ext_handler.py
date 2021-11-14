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

        # self.mapping={"0": "B-ANSWER", "1": "B-HEADER", "2": "B-QUESTION", "3": "E-ANSWER", "4": "E-HEADER", "5": "E-QUESTION", "6": "I-ANSWER", "7": "I-HEADER", "8": "I-QUESTION", "9": "#other", "10": "S-ANSWER", "11": "S-HEADER", "12": "S-QUESTION"}
        # self.mapping={"0": "B-company", "1": "I-company", "2": "B-address", "3": "I-address", "4": "B-total", "5": "I-total", "6": "B-date", "7": "I-date", "8": "#other"}
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
        det_boxes = []
        for row in data:
            val = row.get("recognition") or row.get("body")
            predictions = eval(val.decode())
            df = pd.DataFrame(predictions)
            bboxes = df['bbox'].tolist()
            # boxes = np.array(bboxes).astype('float32')
            # boxes = [[int(1000 * box[0][0]),int(1000 * box[0][1]),int(1000 * box[2][0]),int(1000 * box[2][1])] for box in bboxes]         
            # boxes = []
            # for box in bboxes:
            #     box = np.array(box).astype('float32')
            #     boxes.append([int(1000 * box[0][0]),int(1000 * box[0][1]),int(1000 * box[2][0]),int(1000 * box[2][1]),])
            words = [w[0] for w in df['ocr'].tolist()]   
            features = self.convert_example_to_features(words=words, bboxes=bboxes)
            input_ids.append(torch.tensor(features[0]))
            input_mask.append(torch.tensor(features[1]))
            segment_ids.append(torch.tensor(features[2]))
            token_boxes.append(torch.tensor(features[3]))
            ocr_boxes.append(features[4])
            det_boxes.append(features[5])
        return torch.stack(input_ids).to(self.device), torch.stack(input_mask).to(self.device), torch.stack(segment_ids).to(self.device), torch.stack(token_boxes).to(self.device), ocr_boxes, det_boxes
    
    def merge_text_boxes(self, dt_boxes, rec_res, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5, width_ths = 1.0, add_margin = 0.05):
        dt_boxes = np.asarray(dt_boxes)
        polys = np.empty((len(dt_boxes), 8))
        polys[:, 0] = dt_boxes[:, 0, 0]
        polys[:, 1] = dt_boxes[:, 0, 1]
        polys[:, 2] = dt_boxes[:, 1, 0]
        polys[:, 3] = dt_boxes[:, 1, 1]
        polys[:, 4] = dt_boxes[:, 2, 0]
        polys[:, 5] = dt_boxes[:, 2, 1]
        polys[:, 6] = dt_boxes[:, 3, 0]
        polys[:, 7] = dt_boxes[:, 3, 1]

        (
            horizontal_list,
            free_list_box,
            free_list_text,
            combined_list,
            merged_list_box,
            merged_list_text,
        ) = ([], [], [], [], [], [])

        for i, poly in enumerate(polys):
            slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
            slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
            if max(abs(slope_up), abs(slope_down)) < slope_ths:
                x_max = max([poly[0], poly[2], poly[4], poly[6]])
                x_min = min([poly[0], poly[2], poly[4], poly[6]])
                y_max = max([poly[1], poly[3], poly[5], poly[7]])
                y_min = min([poly[1], poly[3], poly[5], poly[7]])
                horizontal_list.append(
                    [
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        0.5 * (y_min + y_max),
                        y_max - y_min,
                        rec_res[i][0],
                        rec_res[i][1],
                        str(poly),
                    ]
                )
            else:
                height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
                margin = int(1.44 * add_margin * height)
                theta13 = abs(
                    np.arctan(
                        (poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))
                    )
                )
                theta24 = abs(
                    np.arctan(
                        (poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))
                    )
                )
                # do I need to clip minimum, maximum value here?
                x1 = poly[0] - np.cos(theta13) * margin
                y1 = poly[1] - np.sin(theta13) * margin
                x2 = poly[2] + np.cos(theta24) * margin
                y2 = poly[3] - np.sin(theta24) * margin
                x3 = poly[4] + np.cos(theta13) * margin
                y3 = poly[5] + np.sin(theta13) * margin
                x4 = poly[6] - np.cos(theta24) * margin
                y4 = poly[7] + np.sin(theta24) * margin

                free_list_box.append(
                    np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                )
                free_list_text.append(
                    [rec_res[i][0], rec_res[i][1], str(poly), rec_res[i][0]]
                )

        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

        # combine box
        new_box = []
        for poly in horizontal_list:

            if len(new_box) == 0:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                new_box.append(poly)
            else:
                # comparable height and comparable y_center level up to ths*height
                if (
                    abs(np.mean(b_height) - poly[5])
                    < height_ths * np.mean(b_height)
                ) and (
                    abs(np.mean(b_ycenter) - poly[4])
                    < ycenter_ths * np.mean(b_height)
                ):
                    b_height.append(poly[5])
                    b_ycenter.append(poly[4])
                    new_box.append(poly)
                else:
                    b_height = [poly[5]]
                    b_ycenter = [poly[4]]
                    combined_list.append(new_box)
                    new_box = [poly]
        combined_list.append(new_box)

        # merge list use sort again
        for boxes in combined_list:
            if len(boxes) == 1:  # one box per line
                box = boxes[0]
                margin = int(add_margin * min(box[1] - box[0], box[5]))
                _x0 = _x3 = box[0] - margin
                _y0 = _y1 = box[2] - margin
                _x1 = _x2 = box[1] + margin
                _y2 = _y3 = box[3] + margin
                merged_list_box.append(
                    np.array([[_x0, _y0], [_x1, _y1], [_x2, _y2], [_x3, _y3]])
                )
                merged_list_text.append([box[6], box[7], box[8], box[6]])
            else:  # multiple boxes per line
                boxes = sorted(boxes, key=lambda item: item[0])

                merged_box, new_box = [], []
                prev_key = boxes[0][7]
                for box in boxes:
                    curr_key = box[7]
                    if len(new_box) == 0:
                        b_height = [box[5]]
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        if (
                            abs(np.mean(b_height) - box[5])
                            < height_ths * np.mean(b_height)
                        ) and (
                            abs(box[0] - x_max) < width_ths * (box[3] - box[2]) and curr_key == prev_key
                        ):  # merge boxes
                            b_height.append(box[5])
                            x_max = box[1]
                            new_box.append(box)
                        else:
                            b_height = [box[5]]
                            x_max = box[1]
                            merged_box.append(new_box)
                            new_box = [box]
                    prev_key = curr_key
                if len(new_box) > 0:
                    merged_box.append(new_box)

                for mbox in merged_box:
                    if len(mbox) != 1:  # adjacent box in same line
                        # do I need to add margin here?
                        x_min = min(mbox, key=lambda x: x[0])[0]
                        x_max = max(mbox, key=lambda x: x[1])[1]
                        y_min = min(mbox, key=lambda x: x[2])[2]
                        y_max = max(mbox, key=lambda x: x[3])[3]
                        text_comb = (
                            str(mbox[0][6]) if isinstance(mbox[0][6], str) else ""
                        )
                        key_name = mbox[0][7]
                        box_id = str(mbox[0][8])
                        text_id = (
                            str(mbox[0][6]) if isinstance(mbox[0][6], str) else ""
                        )
                        for val in range(len(mbox) - 1):
                            if isinstance(mbox[val + 1][6], str):
                                strin = mbox[val + 1][6]
                            else:
                                strin = ""
                            text_comb += " " + strin
                            # sum_score += mbox[val + 1][7]
                            box_id += "|||" + str(mbox[val + 1][8])
                            text_id += "|||" + strin
                        # avg_score = sum_score / len(mbox)
                        margin = int(add_margin * (y_max - y_min))

                        # merged_list.append([x_min-margin, x_max+margin, y_min-margin, y_max+margin, text_comb, avg_score])
                        _x0 = _x3 = x_min - margin
                        _y0 = _y1 = y_min - margin
                        _x1 = _x2 = x_max + margin
                        _y2 = _y3 = y_max + margin
                        merged_list_box.append(
                            np.array(
                                [[_x0, _y0], [_x1, _y1], [_x2, _y2], [_x3, _y3]]
                            )
                        )
                        merged_list_text.append(
                            [text_comb, key_name, box_id, text_id]
                        )

                    else:  # non adjacent box in same line
                        box = mbox[0]

                        margin = int(add_margin * (box[3] - box[2]))
                        # merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin, box[6], box[7]])
                        _x0 = _x3 = box[0] - margin
                        _y0 = _y1 = box[2] - margin
                        _x1 = _x2 = box[1] + margin
                        _y2 = _y3 = box[3] + margin
                        merged_list_box.append(
                            np.array(
                                [[_x0, _y0], [_x1, _y1], [_x2, _y2], [_x3, _y3]]
                            )
                        )
                        merged_list_text.append([box[6], box[7], box[8], box[6]])

        # may need to check if box is really in image
        return free_list_box, free_list_text, merged_list_box, merged_list_text
    
    def convert_example_to_features(self, words, bboxes, cls_token_box=[0, 0, 0, 0],
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0]):
        boxes = [[int(1000 * box[0][0]),int(1000 * box[0][1]),int(1000 * box[2][0]),int(1000 * box[2][1])] for box in bboxes]                 
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length 
        tokens = []
        token_boxes = []
        ocr_boxes = []
        det_boxes = []
        for word, box, det_box in zip(words, boxes, bboxes):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            det_boxes.extend([det_box] * len(word_tokens))
            ocr_boxes.extend([word] * len(word_tokens))

        # Truncation: account for [CLS] and [SEP] with "- 2". 
        special_tokens_count = 2 
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            det_boxes = det_boxes[: (max_seq_length - special_tokens_count)]
            ocr_boxes = ocr_boxes[:(max_seq_length - special_tokens_count)]

        # add [SEP] token, with corresponding token boxes and actual boxes
        tokens += [tokenizer.sep_token]
        token_boxes += [sep_token_box]
        det_boxes += [sep_token_box]
        ocr_boxes += ['']
        
        segment_ids = [0] * len(tokens)

        # next: [CLS] token
        tokens = [tokenizer.cls_token] + tokens
        token_boxes = [cls_token_box] + token_boxes
        det_boxes = [cls_token_box] + det_boxes
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
        det_boxes += [pad_token_box] * padding_length
        ocr_boxes += [pad_token_box] * padding_length
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        assert len(det_boxes) == max_seq_length
        assert len(ocr_boxes) == max_seq_length
        
        return input_ids, input_mask, segment_ids, token_boxes, ocr_boxes, det_boxes      
    
    def postprocess(self, preds, input_ids, ocrs, bboxes, det_boxes):  
           
        tokenizer = self.tokenizer
        res_list=[]
        res_list2=[]
        # print("Predictions:",preds)
        # print("Input_ids:", input_ids)
        # print("Here"*100,len(preds),len(input_ids))        
        for i in range(len(input_ids)):
            predictions = []
            predictions2 = []
            dt_boxes = []
            text_boxes = []
            # print("Here"*10,len(input_ids[i].squeeze().tolist()))
            res = preds[0][i].argmax(-1).squeeze().tolist()
            # print("Here"*100,len(res),len(input_ids[i].squeeze().tolist()))
            last_ocr = ''
            for id, token_pred, tb, ocr, det_box in zip(input_ids[i].squeeze().tolist(), res, bboxes[i].squeeze().tolist(), ocrs[i], det_boxes[i]):
                if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, 
                                                                        tokenizer.sep_token_id, 
                                                                        tokenizer.pad_token_id]):
                    
                    continue
                else:
                    if ocr != last_ocr:
                        last_ocr= ocr
                    else:
                        continue
                    # json_result= {}
                    if self.mapping:
                        token_pred = self.mapping[str(token_pred)].split('-')[-1]
                    predicted_label = token_pred
                    # rel_box = [b/1000 for b in tb]
                    # json_result["bbox"] = rel_box
                    # json_result["ocr"] = ocr
                    # json_result["key"] = predicted_label   
                    # predictions.append(json_result)
                    # if predicted_label!='#other':
                    #     print(json_result)
                    dt_boxes.append(det_box)
                    text_boxes.append((ocr, predicted_label))
            # res_list.append(predictions)
            free_box, free_text, merged_box, merged_text= self.merge_text_boxes(dt_boxes, text_boxes)
            
            all_boxes = free_box + merged_box
            all_texts = free_text + merged_text
            
            for box,txt in zip(all_boxes, all_texts):
                res={}
                res["bbox"]=[(1 * box[0][0]),(1 * box[0][1]),(1 * box[2][0]),(1 * box[2][1])] 
                res["ocr"]=txt[0]
                res["key"]=txt[1]
                predictions2.append(res)
            res_list2.append(predictions2)
            # print(predictions,'\n',predictions2)
        return res_list2
    
    
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

        input_ids, attention_mask, token_type_ids, bboxes, ocrs, det_boxes = self.preprocess(data)
        model_output = self.inference(input_ids, bboxes, attention_mask, token_type_ids)
        output =  self.postprocess(model_output, input_ids, ocrs, bboxes, det_boxes)
        return output