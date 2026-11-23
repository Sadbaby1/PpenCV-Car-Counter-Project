import cvzone
import math

def draw_bbox_conf_cls(box, frame, model):
    #2 bbox with cv zone
    x1, y1, x2, y2=box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h= x2-x1, y2-y1
    cvzone.cornerRect(frame, (x1, y1, w, h),l=9)

    # add detection confident to the bbox
    conf = math.ceil((box.conf[0]*100))
    cls = int(box.cls[0])
    name = model.names[cls]
    label = f'{name}: {conf}%'
    cvzone.putTextRect(frame,label, ((x1),max(y1-10, 20)), scale=0.9,thickness=1, offset=3)
