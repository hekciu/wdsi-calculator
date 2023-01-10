import os
import cv2
import sklearn
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


class App:
    def __init__(self, img_size_x, img_size_y, numbers_min_spacing, model_train_data_path):
        print("Initializing App")
        self.data_X = []
        self.data_y = []
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.numbers_min_spacing = numbers_min_spacing
        self.model_train_data_path = os.getcwd() + model_train_data_path
        self.signs = ['+', '-', '_', '%', '0', '1', '2',
                      '[', ']']  # '_' means '*' -> multiply operator and '%' means '/' -> divide operator

        self.__load_dataset()
        self.__prepare_model()




        # output = clf.predict(X_test_flatten)
        # plt.imshow(X_test[0])
        # plt.show()
        # print(output[0])

    def __prepare_model(self):
        print("Preparing model")
        X_train, y_train, X_test, y_test = sklearn.model_selection.train_test_split(self.data_X, self.data_y,
                                                                                    test_size=0.25, random_state=52)

        plt.imshow(X_train[0])
        plt.show()
        # self.model = sklearn.ensemble.RandomForestClassifier(random_state=0)
        # X_train_flatten = [item.flatten() for item in X_train]
        # X_test_flatten = [item.flatten() for item in X_test]
        #
        # self.model.fit(X_train_flatten, y_train)
        #
        # huj = self.__check_dark_pixel(X_train[0][:, 4])
        # print(huj)
        # print(X_train[0][:, 4])
        # print(self.__search_columns(X_train[0]))
        # plt.imshow(X_train[0], cmap="Greys")
        # plt.show()

        # clf.fit(X_train_flatten, y_train)

    def compute_equation(self, img):
        pass

    def __parse_equation(self, equ_string):
        pass

    def __load_dataset(self):
        print("Loading train data")
        for sign in self.signs:
            cur_dir_files = [f for f in os.listdir(self.model_train_data_path + '/' + sign)]
            for f in cur_dir_files:
                self.data_X.append(self.read_local_image(self.model_train_data_path + '/' + sign + '/' + f))
                self.data_y.append(sign)

    def show_image(self, img):
        img_resized = cv2.resize(img, (960, 540))
        cv2.imshow('sample image', img_resized)

        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image

    def read_local_image(self, path):
        img = cv2.imread(os.getcwd() + path)
        return img

    def __check_dark_pixel(self, image_column):
        filtered = image_column[image_column > 50]
        return len(filtered) > 0

    def __search_columns(self, image):
        size_x, _ = image.shape
        is_inside_digit = False
        digit_start = None
        digit_end = None
        digit_locations = []

        for i in range(size_x):
            column = image[:, i]
            if self.__check_dark_pixel(column) and not is_inside_digit:
                is_inside_digit = True
                digit_start = i - self.numbers_min_spacing / 2
            if not self.__check_dark_pixel(column) and is_inside_digit:
                is_inside_digit = False
                digit_end = i + self.numbers_min_spacing / 2
                digit_locations.append((digit_start, digit_end))

        return digit_locations
