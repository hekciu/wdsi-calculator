import os
import cv2
import sklearn
import sklearn.model_selection
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import load_config, is_config_valid, parse_and_compute_equation


class App:
    def __init__(self, img_size_x, img_size_y, numbers_min_spacing):
        print("initializing App")

        self.config = load_config()
        if not is_config_valid(self.config):
            return print("config not valid")

        self.data_X = []
        self.data_y = []
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.numbers_min_spacing = numbers_min_spacing
        self.model_train_data_path = os.getcwd() + self.config['MODEL_TRAIN_DATA_PATH']
        self.signs = ['+', '-', '_', '%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      '[', ']']  # '_' means '*' -> multiply operator and '%' means '/' -> divide operator

        if self.config['train_from_scratch']:
            self.__load_dataset()

        self.__prepare_model(self.config['train_from_scratch'])

        if 'test_image_path' in self.config:
            test_img = cv2.imread(os.getcwd() + self.config['test_image_path'], cv2.IMREAD_GRAYSCALE)
            test_equation_arr = self.get_equation_from_image(test_img)
            output = parse_and_compute_equation(test_equation_arr, print_result=True)

    def __prepare_model(self, train_from_scratch=False):
        print("Preparing model")

        if train_from_scratch:
            print('training model')
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.data_X, self.data_y,
                                                                                        test_size=0.3, train_size=0.7)

            X_train_flatten = [item.flatten() for item in X_train]
            X_test_flatten = [item.flatten() for item in X_test]
            self.model = sklearn.ensemble.RandomForestClassifier(random_state=7)
            print("fitting model")
            self.model.fit(X_train_flatten, y_train)
            print("saving model")
            with open(self.config['MODEL_PATH'], "wb") as mod:
                pickle.dump(self.model, mod)
            print('model trained and saved locally')
            print('measuring model accuracy')
            y_test_predicted = self.model.predict(X_test_flatten)
            print(sklearn.metrics.accuracy_score(y_test, y_test_predicted))
        else:
            self.model = pickle.load(open(self.config['MODEL_PATH'], 'rb'))
            print('model downloaded')


    def __load_dataset(self):
        print("Loading train data")
        for sign in self.signs:
            print("Loading data for " + sign)
            cur_dir_files = [f for f in os.listdir(self.model_train_data_path + '\\' + sign)]
            for f in cur_dir_files:
                self.data_X.append(cv2.imread(self.model_train_data_path + '\\' + sign + '\\' + f, cv2.IMREAD_GRAYSCALE))
                self.data_y.append(sign)


    def show_image(self, img):
        img_resized = cv2.resize(img, (960, 540))
        cv2.imshow('sample image', img_resized)

        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image

    def _detect_object(self, img_0):
        # img = cv2.bitwise_not(img_0)
        _, img_th = cv2.threshold(img_0, 110, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        bounding_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([(x, y), (x+w, y+h)])
        for rect in bounding_boxes:
            cv2.rectangle(img_0, rect[0], rect[1], (255, 0, 0), 1)
        cv2.drawContours(img_0, contours, -1, (0, 255, 75), 1)
        cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
        cv2.imshow('Contours', img_th)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
        bounding_boxes = sorted(bounding_boxes, key=lambda l: l[0][0])
        #print(bounding_boxes)
        return bounding_boxes

    def _crop_image(self, img_0, bounds):
        numbers = []
        for rect in bounds:
            (x1, y1), (x2, y2) = rect
            img = img_0[y1:y2, x1:x2]
            img = cv2.resize(img, (28, 28))
            numbers.append(img)
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
        return signs

    def get_equation_from_image(self, img_0):
        bounds = self._detect_object(img_0)
        images = self._crop_image(img_0, bounds)
        return self._get_equation(images)


