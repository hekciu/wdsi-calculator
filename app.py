import os
import cv2
import keras.datasets.mnist
import sklearn
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn.model_selection
import sklearn.ensemble

MODEL_PATH='models/model.pickle'
class App:
    def __init__(self, img_size_x, img_size_y, numbers_min_spacing, model_train_data_path):
        print("Initializing App")
        self.data_X = []
        self.data_y = []
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.numbers_min_spacing = numbers_min_spacing
        self.model_train_data_path = os.getcwd() + model_train_data_path
        self.signs = ['0', '1', '2']
            #['+', '-', '_', '%', '0', '1', '2',
                     # '[', ']']  # '_' means '*' -> multiply operator and '%' means '/' -> divide operator

        self.__load_dataset()
        self.__prepare_model()
        bounds = self._detect_object()
        images = self._crop_image(bounds)
        self._get_equation(images)





    def __prepare_model(self, train_from_scratch=False):
        print("Preparing model")
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        # sklearn.model_selection.train_test_split(self.data_X, self.data_y,
        #                                                                          test_size=0.4, train_size=0.6)
        self.model = sklearn.ensemble.RandomForestClassifier(random_state=0)
        X_train_flatten = [item.flatten() for item in X_train]
        X_test_flatten = [item.flatten() for item in X_test]

        if train_from_scratch:
            print('training model')
            self.model.fit(X_train_flatten, y_train)
            with open(MODEL_PATH, "wb") as mod:
                pickle.dump(self.model, mod)
            print('model trained')
        else:
            self.model = pickle.load(open(MODEL_PATH, 'rb'))
            print('model downloaded')

    def compute_equation(self, img):
        pass

    def __parse_equation(self, equ_string):
        pass

    def __load_dataset(self):
        pass
        # print("Loading train data")
        # for sign in self.signs:
        #     cur_dir_files = [f for f in os.listdir(self.model_train_data_path + '\\' + sign)]
        #     for f in cur_dir_files:
        #         self.data_X.append(self.read_local_image(self.model_train_data_path + '\\' + sign + '\\' + f))
        #         self.data_y.append(sign)
        #self.data_X, self.data_y = keras.datasets.mnist.load_data()


    def show_image(self, img):
        img_resized = cv2.resize(img, (960, 540))
        cv2.imshow('sample image', img_resized)

        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image

    def read_local_image(self, path):
        img = cv2.imread(os.getcwd() + path)
        return img

    def _detect_object(self):
        img_0 = cv2.imread('nr_detection_test.png')
        img = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        img_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print(len(contours))
        bounding_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([(x, y), (x+w, y+h)])
        for rect in bounding_boxes:
            cv2.rectangle(img_0, rect[0], rect[1], (255, 0, 0), 1)
        # cv2.drawContours(img_0, contours, -1, (0, 255, 0), 1)
        # cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
        # cv2.imshow('Contours', img_0)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        bounding_boxes = sorted(bounding_boxes, key=lambda l: l[0][0])
        #print(bounding_boxes)
        return bounding_boxes

    def _crop_image(self, bounds):
        image = cv2.imread('nr_detection_test.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        numbers = []
        for rect in bounds:
            (x1, y1), (x2, y2) = rect
            img = image[y1:y2, x1:x2]
            img = cv2.resize(img, (28, 28))
            numbers.append(img)

        for nr in numbers:
            plt.imshow(nr)
            plt.show()
            if cv2.waitKey(0):
                cv2.destroyAllWindows()

        return numbers
            # height, width = img.shape[:2]
            #
            # blank_image = np.zeros((140, 70, 3), np.uint8)
            # blank_image[:, :] = (255, 255, 255)
            #
            # l_img = blank_image.copy()  # (600, 900, 3)
            #
            # x_offset = 10
            # y_offset = 10
            # # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
            # l_img[y_offset:y_offset + height, x_offset:x_offset + width] = img
            # img = l_img
            #
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def _get_equation(self, images):
        signs = []
        for img in images:
            img_flat = img.flatten()
            img_res = img_flat.reshape(1, -1)
            signs.append(self.model.predict(img_res))
        print(signs)



