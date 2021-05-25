from types import FrameType
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def filter(keypoints, face=None, filter_name=''):
    img = cv2.imread(f'images/{filter_name}', cv2.IMREAD_UNCHANGED)
    mask = img[:, :, 3]
    mask_inv = cv2.bitwise_not(mask)
    img = img[:, :, 0:3]
    # img = None
    # if filter_image.shape[2]==4:
    #     a1 = ~filter_image[:, :, 3]
    #     img = cv2.add(cv2.merge([a1,a1,a1,a1]), filter_image)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    #     cv2.imshow('IMG', img)

    # for i in range(len(keypoints)):
    #     filter_width = keypoints[i]

    filter_width = keypoints[2][0] - keypoints[1][0]
    filter_height = int(filter_width*img.shape[0]/img.shape[1])

    scale_factor = filter_width/img.shape[1]

    # img_orgi = cv2.resize(img,None, fx=scale_factor, fy = scale_factor, interpolation=cv2.INTER_AREA)
    img_orgi = cv2.resize(img, (filter_width, filter_height), interpolation=cv2.INTER_AREA)
    width, height = img_orgi.shape[1], img_orgi.shape[0]

    # mask = cv2.resize(mask,None, fx=scale_factor, fy = scale_factor, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (filter_width, filter_height), interpolation=cv2.INTER_AREA)
    # mask_inv = cv2.resize(mask_inv,None, fx=scale_factor, fy = scale_factor, interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(mask_inv, (filter_width, filter_height), interpolation=cv2.INTER_AREA)
    

    # x1 = int((keypoints[2][0] - keypoints[1][0])/2 - width/2)
    # x2 = x1+width
    
    # y1 = abs(int((keypoints[2][1] - keypoints[1][1])/2 - height/2))
    # y2 = y1+height

    # roi = face[keypoints[1][1]:keypoints[2][1]+y1, keypoints[1][0]:keypoints[2][0]]

    y1 = int(keypoints[2][1])-15
    y2 = int(y1+filter_height)
    x1 = int(keypoints[1][0])
    x2 = int(x1+filter_width)
    roi = face[y1:y2, x1:x2]
    cv2.imshow('Test', roi)

    # alpha_fil = np.expand_dims(sg[:, :, 3]/255.0, axis=-1)
    alpha_fil = float(0.5)
    alpha_face = 1.0 - alpha_fil

    # new = alpha_fil * sg[:, :, :3]
    print(roi.shape, mask_inv.shape)
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_fg = cv2.bitwise_and(img_orgi, img_orgi, mask=mask)

    face[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    # print(alpha_face.shape, face.shape, alpha_fil.shape, new.shape)
    
    
    # face[keypoints[1][1]:keypoints[2][1]+y1, keypoints[1][0]:keypoints[2][0]] = (alpha_fil * sg[:, :, :3] + 
    #                     face[keypoints[1][1]:keypoints[2][1]+y1, keypoints[1][0]:keypoints[2][0]])

    return face

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape
        if results.multi_face_landmarks:
            keypoints = []
            
            for face_landmarks in results.multi_face_landmarks:
                forehead_x, forehead_y = int(face_landmarks.landmark[10].x*w), int(face_landmarks.landmark[10].y*h)
                # cv2.circle(image, (forehead_x, forehead_y), 5, (255, 0, 0), 2)
                keypoints.append([forehead_x, forehead_y])

                leftear_x, leftear_y = int(face_landmarks.landmark[162].x*w), int(face_landmarks.landmark[162].y*h)
                # cv2.circle(image, (leftear_x, leftear_y), 5, (255, 0, 0), 2)
                keypoints.append([leftear_x, leftear_y])

                # Prev: right 454, left 234
                rightear_x, rightear_y = int(face_landmarks.landmark[389].x*w), int(face_landmarks.landmark[389].y*h)
                # cv2.circle(image, (rightear_x, rightear_y), 5, (255, 0, 0), 2)
                keypoints.append([rightear_x, rightear_y])

                chin_x, chin_y = int(face_landmarks.landmark[152].x*w), int(face_landmarks.landmark[152].y*h)
                # cv2.circle(image, (chin_x, chin_y), 5, (255, 0, 0), 2)
                keypoints.append([chin_x, chin_y])

                nose_top_x, nose_top_y = int(face_landmarks.landmark[6].x*w), int(face_landmarks.landmark[6].y*h)
                # nose_bottom_x, nose_bottom_y = int(face_landmarks.landmark[2].x*w), int(face_landmarks.landmark[2].y*h)
                nose_left_x, nose_left_y = int(face_landmarks.landmark[129].x*w), int(face_landmarks.landmark[129].y*h)
                nose_right_x, nose_right_y = int(face_landmarks.landmark[358].x*w), int(face_landmarks.landmark[358].y*h)

                pts = np.array([
                    [nose_top_x, nose_top_y], 
                    # [nose_bottom_x, nose_bottom_y], 
                    [nose_left_x, nose_left_y], [nose_right_x, nose_right_y]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))

                # cv2.polylines(image, [pts], True, (0, 255, 255), 2)

                glass = filter(keypoints, image, filter_name='glass.png')
                
                # print(x, y, z)
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACE_CONNECTIONS,
                #     landmark_drawing_spec=drawing_spec,
                #     connection_drawing_spec=drawing_spec)
            cv2.imshow('MediaPipe FaceMesh', image)
            cv2.imshow('Glass', glass)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

