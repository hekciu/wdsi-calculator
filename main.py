from app import App

app = App(28, 28, 2, '/model_train_data')

if __name__ == '__main__':
    print("Hello World")
    app.read_local_image("/data/ewelina.jpg")
