import time
import numpy as np
from PIL import Image
import os 
import sys 
import json
sys.path.append(os.getcwd())
from deploy.detector import Detector, get_detector

@profile
def time_detector_infer(objs):
    Detector =  get_detector()
    layout = Detector.infer(objs)
    with open("layout.json", "w") as outfile:
        json.dump(layout, outfile, indent=4)

if __name__ == '__main__':
    paths = ["deploy/test/scanned-{}.png".format(i) for i in range(10)]
    # paths = ["deploy/test/scanned-{}.png".format(2)]
    imgs = [np.array(Image.open(p)) for p in paths]

    for _ in range(5):
        start_time = time.perf_counter()
        time_detector_infer(imgs)
        print("total execution time -->{:.5f} s".format(time.perf_counter() - start_time))
    #run "kernprof -lv .\deploy\test\test_time_inference.py"
    