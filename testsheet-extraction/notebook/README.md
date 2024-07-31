# Detech-QA-from-ExerSheetPDF
 
![fig](../assets/overview.png)

- detect question from scanned pdf file into text

## Steps:

1. Element Detection
* I assume that an exam paper is built by 8 main components:

	- heading
	- question
	- subquestion
	- auxillary_text
	- choice
	- blank
	- image
	- table

* Basicly, I can utilize Yolo to detect them. To train yolo, reading the [training-yolo.ipynb](training-yolo.ipynb)
* Note: dataset, model I used in .ipynb files are linked to my drive, therefore u should change the paths in all files.

2. OCR

- after receiving detection results, we can apply ocr model on them. 
- In this work, for math formulas, I recomand use mathpix ocr api.

3. Post processing

- because I detect seperately elements, so I need to group elements belonging to same question. 

4. MORE
- for more, read [Pipeline](Pipeline_YOLOxMathPix.ipynb)
- slide: [summary](Yolov8xOCR.pdf)
				