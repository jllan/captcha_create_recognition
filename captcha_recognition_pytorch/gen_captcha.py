from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import pickle
import os

# TARGET_HEIGHT=48
# TARGET_WIDTH=128
MAXLEN = 6
TARGET_HEIGHT=70
TARGET_WIDTH=160
# MAXLEN = 4

number = [str(i) for i in range(10)]
alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
Alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]
# alphabet.remove('o')
# Alphabet.remove('O')
charset = number+alphabet+Alphabet


def random_chars(charset, nb_chars):
    return [np.random.choice(charset) for i in range(nb_chars)]


def gen_captcha(charset, nb_chars=None, font=None):
    buffer_index = 1000
    buffer_size = 1000
    nb_set = np.zeros(buffer_size)

    generator = ImageCaptcha(width=TARGET_WIDTH, height=TARGET_HEIGHT, fonts=font)

    while True:
        if buffer_index == buffer_size:
            nb_set = np.random.randint(3, MAXLEN+1, buffer_size) if not nb_chars else np.array([nb_chars] * buffer_size) # 一次性生成buffer_size个值，提高程序性能
            buffer_index = 0
        captcha_text = ''.join(random_chars(charset, nb_set[buffer_index]))
        buffer_index += 1
        # img_text = ' '*np.random.randint(0, MAXLEN+1-len(captcha_text))*2+captcha_text #用空格模拟偏移
        captcha = generator.generate(captcha_text)
        captcha_image = Image.open(captcha).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.ANTIALIAS)
        # generator.write(captcha_text, os.path.join('data', captcha_text+'.jpg'))  # 写到文件
        captcha_array = np.array(captcha_image)
        yield captcha_array, captcha_text


def convert_to_npz(num=None, captcha_generator=None, pic_dir=None, is_encoded=True, is_with_tags=True):
    vocab = charset[:]
    if is_encoded:
        vocab += [' ']
    if is_with_tags:
        id2token = {k+1:v for k,v in enumerate(vocab)}
        id2token[0] = '^'
        id2token[len(vocab)+1]='$'
    else:
        id2token = dict(enumerate(vocab))

    token2id = {v:k for k,v in id2token.items()}

    vocab_dict ={"id2token": id2token, "token2id": token2id}
    with open("data/captcha.vocab_dict", "wb") as dict_file:
        pickle.dump(vocab_dict, dict_file)

    fn = "data/captcha.npz"
    print("Writing ", fn)

    text_buffer = []
    if not captcha_generator:
        pics = os.listdir(pic_dir)
        img_buffer = np.zeros((len(pics), TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        for i, pic in enumerate(pics):
            title = os.path.splitext(os.path.basename(pic))[0].lower()
            img = Image.open(os.path.join(pic_dir, pic)).convert('RGB')
            img = np.array(img)
            img_buffer[i] = img
            if is_with_tags:
                title = ("^" + title + "$")
            if is_encoded:
                text_buffer.append([token2id[i] for i in title.ljust(MAXLEN + 2 * is_with_tags)])
            else:
                text_buffer.append(title)
    else:
        img_buffer = np.zeros((num, TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        for i in range(num):
            x, y = next(captcha_generator)
            img_buffer[i] = x
            if is_with_tags:
                y = ("^"+y+"$")
            if is_encoded:
                text_buffer.append([token2id[i] for i in y.ljust(MAXLEN+2*is_with_tags)])
            else:
                text_buffer.append(y)

    np.savez(fn, img=img_buffer, text=text_buffer)
    return vocab_dict, img_buffer, text_buffer


if __name__ == '__main__':
    # 生成验证码
    # vocab_dict, img, text = convert_to_npz(num=4000, captcha_generator=gen_captcha(charset, 4),
    #                                        is_encoded=True, is_with_tags=True)

    # 从本地读取验证码
    vocab_dict, img, text = convert_to_npz(pic_dir='/home/jlan/WorkSpace/cv/dataset/court',
                                           is_encoded=True, is_with_tags=True)
