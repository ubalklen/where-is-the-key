import os
import sys
import numpy as np
from PIL import Image, ImageChops, ImageOps
from keras.models import Model, load_model
from tqdm import tqdm

def prepare_input(x):
    x = np.array(x)
    x = x / 255
    x = x.flatten()
    return np.asarray([x])

if sys.argv[1]:
    #supress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #open the model
    model = load_model('model.h5')

    #open the image
    img = Image.open(sys.argv[1])
    (img_width, img_height) = img.size
    rangex = int(img_width/10)
    rangey = int(img_height/10)
    
    #scan the image to find the key
    key_path = os.path.dirname(os.path.abspath(__file__)) + '/' + os.path.basename(img.filename) + ' key'
    if not os.path.exists(key_path):
        os.makedirs(key_path)

    key_boxes = [(0, 0, 0, 0)]
    for y in tqdm(range(rangey)):
        for x in range(rangex):
            box = (x * 10, y * 10, x * 10 + 40, y * 10 + 40)
            slice_bit = img.crop(box)
            slice_bit_array = prepare_input(slice_bit)
            pred = model.predict(slice_bit_array)
            prob_key = pred[0][1]
            if prob_key > 0.999:
                slice_bit.save(key_path + '/' + str(box) + ' ' + str(prob_key) + '.jpg')
                key_boxes.append(box)

    if key_boxes == [(0, 0, 0, 0)]:
        print('\n\nNo key found in ' + os.path.basename(img.filename))
    else:
        print('\n\nKey found! Check the "' + os.path.basename(img.filename) + ' key" folder')
