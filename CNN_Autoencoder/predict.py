from keras.models import load_model
from CNN_Autoencoder.data_gen import prepare_mnist_data
import numpy as np
import os
import cv2

from keras.models import Model
def predict_accuracy(x_batch, y_batch, model, output_dir):
    preds = model.predict(x_batch, verbose=0)
    batch_size = len(x_batch)
    count = 0
    for (y, pred) in zip(y_batch[0], preds[0]):
        if np.argmax(y) == np.argmax(pred):
            count += 1
    print('acc ', (count/batch_size))

    os.makedirs(output_dir, exist_ok=True)
    for (i, (y, pred)) in enumerate(zip(y_batch[1], preds[1])):
        org_img = np.array((255 - y) * 255, dtype=np.uint8)
        pred_img = np.array((255 - pred) * 255, dtype=np.uint8)

        output_img = cv2.hconcat([org_img, pred_img])
        output_name = output_dir + str(i).zfill(2) + '.png'
        cv2.imwrite(output_name, output_img)


if __name__ == '__main__':
    model_path = 'model/MNIST_test_epoch_01_model.hdf5'
    model = load_model(model_path)
    output_dir = 'output_image/'

    (_, _), (x_test, y_test) = prepare_mnist_data()
   
    print(model.layers)
    intermediante_layer_model = Model(inputs=model.input, outputs=model.get_layer("max_pooling2d_2").output)
    y = intermediante_layer_model.predict(x_test[0:10])
    print(y)
    print(y.shape)
    predict_accuracy(x_test, y_test, model, output_dir)


