import tensorflow as tf
print(tf.__version__)

import tarfile  #압축파일
import urllib.request

import os

# 모델 다운로드하고 압축 푸는 코드
MODEL_DATE = '20200711'

#코코 2017년 데이터로 학습시킨 것.
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8' #ssd의 앞부분은 모바일넷으로 하고.
#이름을 이렇게 해놨기 때문에 이 데이터셋이 어떤 건지 대충 알 수 있다.
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'

MODELS_DIR = 'data/models'

#모델 다운로드 받을 수 있는 곳이 있습니다. -> tf zoo
#이 경로에 있는걸 다운로드 받을 겁니다.
#http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz 여기서 가져왔다.
#나중에 다른 모델 사용할 일 생기면 여기서.
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME

PATH_TO_MODEL_TAR = os.path.join('data/models', MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join('data/models', os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join('data/models', os.path.join(MODEL_NAME, 'pipeline.config'))


#모델 받은 거 압축파일 푸는 코드. 교수님 올린 코득 그냥 복붙. 로직이 아니니 그냥 복붙해서 써라
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

#레이블 다운로드 받는 코드. 교수님 올린 코득 그냥 복붙. 로직이 아니니 그냥 복붙해서 써라.
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')

##모델 로딩!
#이게 이번에 설치한 라이브러리
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

#나중에 혹시 gqu를 활용해야하면
#이러면 gpu를 가져온다. gpu가 여러개일 수도 있다.
#GPU의 Dymamic Memory Allocation을 활성화 시키는 코드
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# config 로드하고, 모델 빌드
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
#컨픽 가지고 와서...
#이게 모델 컨픽이다.
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

#Restore Checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
#파일을 하나 만들겠다.
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial() #예제에 있던 거 그대로 가져온 것이다. 교수님이 만든 거 아니다. 그냥 순서가 이렇다고 나와있다.

#골뱅이는 어노테이션. jwt인증토큰할 때 쓰는 거 했엇다.
#여기선 텐서플로우의 펑션이라는 뜻이다.
#이미지를 주면 예측해주는 함수이다. 이게 끝이다. 나머지는 우리가 로직짜주는 거.
@tf.function
def detect_fn(image):
    """Detect Objects in Image.""" #함수 정의 바로 밑에다가 이렇게 """ 하고 글 쓰면, 이건 함수에 대한 주석이다. 파라미터 같은 거.
    image, shapes = detection_model.preprocess(image)  #프리프로세스 : 예측 전 가공
    prediction_dict = detection_model.predict(image,shapes) #예측
    detections = detection_model.postprocess(prediction_dict,shapes) #포스트 프로세스 : 예측후 가공
    return detections, prediction_dict, tf.reshape(shapes, [-1])
#여러 모델들 보고, 상황에 맞는 모델을 가져다 쓰면 되겠죠?


## 레이블 맵 데이터 로딩 (Load label map data)
#어떤 레이블인지 인덱스 있어야죠.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

print(category_index)



import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)

while True:
    ret, image_np = cap.read()
    #(1, x, x, 3) 이렇게 만들 겁니다. 여러분 카메라 해상도를 모르니까 xx 로 쓴 것입니다.
    image_np_expended = np.expand_dims(image_np, axis = 0)
    #넘파이를 텐서로 바꿔서 만들었던 함수호출 해주면 된다.
    #그럼 결과 3개를 리턴하니까 그걸 변수에 저장한 것.
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype = tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    
    # print("야 여기봐", detections)

    #mscoco_label_map.pbtxt 파일을 보면, id가 1부터 시작하니까
    #   offset을 1로 만들어준다. 그래야 우리 계산이 편하니까.
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    #비주얼라이징 유틸들을 말함. 위에 네모박스 나오고 이런거.
    #함수 이름 엄청 명확하다. 박스랑 라벨을 이미지 배열 위에.
    #텐서로 되어있는 건 넘파이로 바꿔줄 것.
    #min_score_threshold : 클래스가 20개 있다고 치자. 어느 정도 수치가 높게 나온(0.6 0.7 이 정도) 것들만 보겠다.
    viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'][0].numpy(),
                    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                    detections['detection_scores'][0].numpy(),
                    category_index,
                    use_normalized_coordinates = True,
                    max_boxes_to_draw = 200,
                    min_score_thresh = 0.6,
                    agnostic_mode = False)


    # cv2.imshow("object detection", image_np_with_detections)
    #위와 같이 써도 되고, 아래와같이 써도 된다. 하지만 굳이 리사이즈 해보겠다.
    #관제 시스템의 경우 디스플레이 크면 거기에 맞게 조절해서 보여주도록 하는 코드.
    cv2.imshow("object detection", cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()