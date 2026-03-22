from ultralytics import YOLO
import cv2
import easyocr
import re
from collections import defaultdict,deque



plate_history = defaultdict(lambda: deque(maxlen=10)) # last 10 predictions per box
final_plate = {}

def get_box_id(x1,y1,x2,y2):
    # use rounded coordinates as a psudo id
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id,new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        # Majority Vote
        most_common = max(set(plate_history[box_id]),key = plate_history[box_id].count)
        final_plate[box_id] = most_common
    return final_plate.get(box_id,"")


def validate_plate_format(ocr_text):
    mapping_num_to_alpha = {"0":"O", "1":"I", "2":"Z", "5":"S", "8":"B"}
    mapping_alpha_to_num = {"O":"0","I":"1","Z":"2","S":"5","B":"8"}
    # Remove space after uppercase
    ocr_text = ocr_text.upper().replace(" ","")
    if len(ocr_text) != 7:
        return "" # Invalid Plate Format
    validated_plate = []
    for i ,ch in enumerate(ocr_text):
        if i>3: # alphabet position
            if ch.isdigit() and ch in mapping_num_to_alpha:
                validated_plate.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                validated_plate.append(ch)
            else:
                return "" # Invalid Char
        else: # Numeric Position
            if ch.isalpha() and ch in mapping_alpha_to_num:
                validated_plate.append(mapping_num_to_alpha[ch])
            elif ch.isdigit():
                validated_plate.append(ch)
            else:
                return "" # Invalid Char
            
    return "".join(validated_plate)

def recognize_plate(cropped_plate):
    if cropped_plate.size == 0:
        return ""
    # Preprocess for OCR
    gray_img = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized_plate = cv2.resize(th,None,fx =2, fy = 2,interpolation=cv2.INTER_CUBIC)
    try:
        allow_chars = 'ABCDEFGHIJKLMNOPQRSTWUVXYZ1234567890'
        ocr_result = reader.readtext(resized_plate,detail = 0,allowlist=allow_chars)
        if len(ocr_result)> 0:
            candidate = validate_plate_format(ocr_result[0])
            print(f"Candidate Found: {candidate}")
            if candidate and plate_pattern.match(candidate):
                print(f"Valid Plate Found: {candidate}")
                return candidate
    except Exception as e:
        print(f"Exception occurred: {e}")
        pass

    return ""


# Load fine tuned YOLO model
model = YOLO("best.pt")

# Load Reader

## This is for MacOS SSL issue, you can remove it if not needed
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
reader = easyocr.Reader(lang_list=['en'])

# Plate Pattern
plate_pattern = re.compile(r"^[0-9]{4}[A-Z]{3}$")


# Video Inference

input_video = "./video/video.mp4"
output_video = "license.mp4"
cap = cv2.VideoCapture(input_video)
# If you want to save the output video, uncomment the following lines and make sure to install the required codecs for mp4v
## fourcc = cv2.VideoWriter_fourcc(*"mp4v")
## out = cv2.VideoWriter(output_video,fourcc,cap.get(cv2.CAP_PROP_FPS),(int(cap.get(3)),int(cap.get(4))))

CONF_TH = 0.3

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    results = model(frame,verbose = False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < CONF_TH:
                continue
            x1,y1,x2,y2 = map(int,box.xyxy.cpu().numpy()[0])
            cropped_plate = frame[y1:y2, x1:x2]

            # OCR with correction
            print(f"Cropped Image Size:{cropped_plate.size}")
            text = recognize_plate(cropped_plate)

            # Stabilize text using history
            box_id = get_box_id(x1,y1,x2,y2)
            stable_text = get_stable_plate(box_id,text)

            # Draw rectangle around the plate
            cv2.rectangle(frame,(x1, y1),(x2, y2),(0,255,0), 3)
            
            # Overlay zoom in plate above the detected plate
            if cropped_plate.size > 0 :
                overlay_w, overlay_h = 400,150
                resized_plate = cv2.resize(cropped_plate,(overlay_w,overlay_h))
                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2,ox2 = oy1 + overlay_h, ox1 + overlay_w 

                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2,ox1:ox2] = resized_plate
                    # Show stable text above the overlay
                    print(f"Box ID: {box_id}, New Text: {text}, Stable Text: {stable_text}")
                    if stable_text:
                        cv2.putText(frame, stable_text,
                                    (ox1, oy1 - 20),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6) # black outline
                        cv2.putText(frame, stable_text, 
                                    (ox1, oy1 - 20),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3) # white text
    #out.write(frame)                                        
    cv2.imshow("Annotated Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#out.release()
cap.release()
cv2.destroyAllWindows()
print(f"Video saved as {output_video}")



