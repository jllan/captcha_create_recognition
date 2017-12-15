from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random,numpy,string


table  =  []
for  i  in  range( 256 ):
    table.append( i * 1.97 )

def create_captcha_image(chars):
    # image_array = numpy.ones((width, length, 3), dtype=numpy.uint8) * 255
    # image = Image.fromarray(image_array)
    background = random_color(255, 255)
    image = Image.new('RGB', (160, 70), background)
    draw = ImageDraw.Draw(image)

    images = []
    for c in chars:
        images.append(_draw_character(draw, c))

    text_width = sum([im.size[1] for im in images])
    width = max(text_width, 160)
    image = image.resize((width, 70))
    average = int(text_width / len(chars))
    rand = int(0.25 * average)
    offset = int(average * 0.1)
    for im in images:
        w, h = im.size
        mask = im.convert('L').point(table)
        image.paste(im, (offset, int((70 - h) / 2)), mask)
        offset = offset + h + random.randint(-rand, 0)

    if width > 70:
        image = image.resize((160, 70))

    image = image.filter(ImageFilter.SMOOTH)
    image = create_noise_curve(image)
    image.save('captch.png')  # 保存验证码图片
    return image


def _draw_character(draw, c):
    """生成透明字符图"""
    font = ImageFont.truetype('C:\\Windows\\Fonts\\Microsoft YaHei UI\\msyh.ttc', 45)
    w, h = draw.textsize(c, font=font)
    dx = random.randint(0, 4)
    dy = random.randint(0, 6)
    im = Image.new('RGBA', (w + dx, h + dy))
    ImageDraw.Draw(im).text((dx, dy), c, font=font, fill=random_color(0, 255))
    # im = rotate(im)
    im = wrap(im, w, h)
    return im

def rotate(img):
    """图片旋转"""
    img = img.crop(img.getbbox())
    img = img.rotate(random.uniform(-30, 30), Image.BILINEAR)
    return img


def wrap(img, w, h):
    """图片扭曲"""
    dx = w * random.uniform(0.3, 0.3)
    dy = h * random.uniform(0.3, 0.3)
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    x2 = int(random.uniform(-dx, dx))
    y2 = int(random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    data = (
        x1, y1,
        -x1, h2 - y2,
        w2 + x2, h2 + y2,
        w2 - x2, -y1,
    )
    img = img.resize((w2, h2))
    img = img.transform((w, h), Image.QUAD, data)
    return img

def create_noise_curve(image):
    w, h = image.size
    x1 = random.randint(0, int(w / 5))
    x2 = random.randint(w - int(w / 5), w)
    y1 = random.randint(int(h / 5), h - int(h / 5))
    y2 = random.randint(y1, h - int(h / 5))
    points = [x1, y1, x2, y2]
    end = random.randint(160, 200)
    start = random.randint(0, 20)
    ImageDraw.Draw(image).arc(points, start, end, fill=random_color(0, 255))
    return image


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


s = string.ascii_letters+string.digits
s = random.sample(s, 4)
print(s)
create_captcha_image(s)
