#Thirst Library
from ultralytics import YOLO
import multiprocessing as mp
import concurrent.futures

# from deploy.utils import Config, subprocess_ocr_request, group_element, REVERSE_MAP
from utils import Config, subprocess_ocr_request, group_element, REVERSE_MAP


################################################
#               RESQUEST
#################################################
class DetectorSJ(object):
    def __init__(self):
        self.cfg = Config()
        self.model = YOLO(self.cfg.store_path)

    # predict from image
    def ready(self):
        return str(type(self.model))
   
    # @profile
    def infer(self, inputs):
        """
        inputs: upload file from http requestion

        Return --> list(dict) : list of question objects with their child elements
        """
        #Prediction
        detection_results = self.model.predict(inputs, save=False, imgsz=(640, 640), conf=self.cfg.threshold, device=self.cfg.device)

        #OCR                      
        parent_connections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for page_index, results in enumerate(detection_results):
                for box in results.boxes:
                    parent_conn, child_conn = mp.Pipe()
                    parent_connections += [parent_conn]
                            
                    executor.submit(subprocess_ocr_request, (page_index, int(box.xyxy[0][1])), REVERSE_MAP[int(box.cls)],
                                detection_results[page_index].orig_img[int(box.xyxy[0][1]): int(box.xyxy[0][3]), int(box.xyxy[0][0]): int(box.xyxy[0][2])],
                                self.cfg.matpix_id, self.cfg.matpix_key, self.cfg.s3_bucket_name,
                                child_conn)
        elements = []
        for parent_connection in parent_connections:
            elements += [parent_connection.recv()]

        #Postprocesing
        elements = sorted(elements, key=lambda x: x["idx"])
        questions = group_element(elements)

        return questions     

