from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from captcha_create import create_captcha_image
import string

characters = string.digits + string.ascii_letters
width, height, n_len, n_class = 160, 70, 4, len(characters)


def gen(batch_size=16):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for _ in range(n_len)]
    # generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join(random.sample(characters, 4))
            # print('random_str: ', random_str)
            # X[i] = generator.generate_image(random_str)
            X[i] = create_captcha_image(random_str, width=width, height=height)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


def gen_model():
    input_tensor = Input(shape=(height, width, 3))
    x = input_tensor
    print(x)
    for i in range(4):
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = MaxPooling2D((2, 2), padding='same', data_format="channels_last")(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    model = Model(inputs=input_tensor, outputs=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def show_model(model):
    from keras.utils.vis_utils import plot_model as plot
    from IPython.display import Image

    plot(model, to_file="model.png", show_shapes=True)
    Image('model.png')


def train(model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    filepath = './checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(filepath, save_best_only=True, save_weights_only=True)

    model.fit_generator(gen(), steps_per_epoch=1600, epochs=5,
                        workers=4, use_multiprocessing=True,
                        validation_data=gen(), validation_steps=40,
                        callbacks=[early_stopping, model_checkpoint])
    model.save('captcha_cnn2.h5')


def test(model):
    X, y = next(gen(1))
    print(X.shape)
    y_pred = model.predict(X)
    plt.title('real: %s\npred:%s' % (decode(y), decode(y_pred)))
    plt.imshow(X[0], cmap='gray')
    plt.show()


def test_from_file(model, pic):
    title = os.path.splitext(os.path.basename(pic))[0].lower()
    img =Image.open(pic).convert('RGB')
    X = np.array(img)
    X = X.reshape([1]+list(X.shape))
    y_pred = model.predict(X)
    print('real: %s\tpred:%s' % (title, decode(y_pred)))


def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = next(generator)
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=2).T
        y_true = np.argmax(y, axis=2).T
        batch_acc += np.mean(list(map(np.array_equal, y_true, y_pred)))
    return batch_acc / batch_num


def main():
    model = load_model('captcha_cnn2.h5')
    pic_dir = '/home/jlan/WorkSpace/cv/dataset/court'
    pics = os.listdir(pic_dir)
    for pic in pics:
        test_from_file(model, os.path.join(pic_dir, pic))


if __name__ == '__main__':
    # model = gen_model()
    # train(model)
    # print(evaluate(model))
    # model = load_model('captcha_cnn2.h5')
    # print(evaluate(model))
    # test(model)
    main()