# PlateNumberDetection
The purpose of the project is detecting particular plate number . (You can adjust plate pattern if needed).

## How It Works
YOLO v8 pretrained model is fine tuned using Roboflow . Since YOLO v8 model does not have plate number as a class.
After training , The model was detecting plate itself not the text . To do that , easyOCR is used ;however, there were still some issues . Sometimes it was confused number and character since they look so similar . This issue is fixed using hard coded mapping . The another issue was that model sometimes was detecting the correct plate text at the upcoming frame and sometimes eary frames. This issue is fixed just looking at 10 frames and decide which text is the most common one among 10 frames.

## Results

https://github.com/user-attachments/assets/673b18dc-f1bc-4a70-8439-279d62374db1

## Resources 
- https://www.youtube.com/watch?v=9KnARRqbkzY&t=3712s
- https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
