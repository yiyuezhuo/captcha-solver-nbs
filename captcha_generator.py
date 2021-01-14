import numpy as np
import cv2
import random
from scipy.stats import multivariate_normal
import string

string_range = string.ascii_letters + string.digits

def sample_text():
    return ''.join(random.choices(string_range, k=4))

base_size = (80, 215, 3)

# The result is obtained by two components mixture gaussian model

background_mean = np.array([209.64227612, 211.76968813, 210.51193557])
background_cov = np.array([
    [ 806.25271058,  -13.49901808,   58.50646508],
    [ -13.49901808,  804.24909933,   72.22467487],
    [  58.50646508,   72.22467487,  877.51980276]
])
text_mean = np.array([ 92.78387704,  94.42889647,  94.50165166])
text_cov = np.array([
    [1915.18389446,  314.3247593 ,  436.24635245],
    [ 314.3247593 , 1738.59386039,  253.08728163],
    [ 436.24635245,  253.08728163, 1896.97596315]
])

background_rgb_free_dist = multivariate_normal(background_mean, background_cov) # cache Cholesky result
text_rgb_free_dist = multivariate_normal(text_mean, text_cov)

font_list = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_ITALIC]
    # cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_ITALIC]


def sample_rgb_dist(dist, size=None):
    res = dist.rvs() if size is None else dist.rvs(size)
    return res.clip(0, 255).astype(np.uint8)


def random_background():
    background_color = sample_rgb_dist(background_rgb_free_dist)
    return np.full(base_size, background_color)


def random_base(background=None, *, text=None, font=None):
    if background is None:
        background = random_background()
    if text is None:
        text = sample_text()
    # text_color = tuple(sample_rgb_dist(text_rgb_free_dist))
    text_color = sample_rgb_dist(text_rgb_free_dist)

    org_x_all_offset = np.random.randint(-4, 4)
    org_y_all_offset = np.random.randint(-2, 2)
    for idx, c in enumerate(text):
        org_x = 57 + idx*26 + np.random.randint(-2, 2) + org_x_all_offset
        org_y = 44 + np.random.randint(-2, 2) + org_y_all_offset
        org = (org_x, org_y)
        fontScale = np.random.random()*0.7 + 0.9
        # thickness = np.random.random()*1.5 + 0.5
        if font is None:
            font = random.choice(font_list)
        thickness = np.random.randint(2, 5) # Argument 'thickness' is required to be an integer
        # cv2.putText(background, c, org, font, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(background, c, org, font, fontScale, text_color.tolist(), thickness, cv2.LINE_AA)
    return background


def affine_transform(image=None):
    if image is None:
        image = random_base()
    h, w = image.shape[:2]
    a = 1/3
    b = 2/3
    c = h/80 # 80 is selected arbitrarily and it can be adjusted to increase or decrease thee effoect of this transform
    d = w/80
    p1 = np.array([[h * a, w * a], [h * a, w * b], [h * b, w * b]], dtype=np.float32)
    d = np.c_[(np.random.random(3)-0.5)*c, (np.random.random(3)-0.5)*d].astype(np.float32)
    p2 = p1 + d
    T = cv2.getAffineTransform(p1, p2)
    image = cv2.warpAffine(image, T, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # import pdb;pdb.set_trace()
    return image


def deform(image=None, *, alpha=50, sigma=6, interpolation=cv2.INTER_LINEAR):
    if image is None:
        image = affine_transform()
    shape = image.shape[:2]
    
    blur_size = int(4*sigma) | 1
    dx = alpha * cv2.GaussianBlur(np.random.random(shape) * 2 - 1, ksize=(blur_size, blur_size), sigmaX=sigma)
    dy = alpha * cv2.GaussianBlur(np.random.random(shape) * 2 - 1, ksize=(blur_size, blur_size), sigmaX=sigma)
    
    dz = np.zeros_like(dx)

    x, y= np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    px = (x + dx).clip(1, image.shape[1]-1).astype(np.float32) # emulate reflect pad
    py = (y + dy).clip(1, image.shape[0]-1).astype(np.float32)

    return cv2.remap(image, px, py, interpolation=interpolation)


def max_distance(image):
    return np.sqrt(image.shape[0]**2 + image.shape[1]**2) * (np.random.random() * 0.5 + 0.5)


def get_transform_matrix(angle, scale_x, scale_y, pos_x, pos_y):
    scale_mat = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    rot_mat = np.zeros([3, 3])
    rot_mat[:2,:2] = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rot_mat[2, 2] = 1.
    
    translate_mat = np.array([
        [1, 0, pos_x],
        [0, 1, pos_y],
        [0, 0, 1]
    ])
    
    return translate_mat @ rot_mat @ scale_mat


def draw_transform_matrix(image):
    angle = (np.random.random() * 0.5 -0.25) * np.pi
    scale_x = max_distance(image) * (np.random.random() * 0.5 + 0.5)
    scale_y = 1.
    pos_x = (np.random.random() * 0.8 + 0.1) * image.shape[1]
    pos_y = (np.random.random() * 0.8 + 0.1) * image.shape[0]
    # print(angle, scale, pos_x, pos_y)
    return get_transform_matrix(angle, scale_x, scale_y, pos_x, pos_y)


def box_idxy(image, idxy):
    """
    idxy:
    [1 2 3 4
     5 6 7 8
     ...]
    """
    h, w = image.shape[:2]
    idxy = idxy[:, (idxy[0, :] < w) & (idxy[0,:] >= 0) & (idxy[1, :] < h) & (idxy[1, :] >= 0)]
    idxy = np.unique(idxy, axis=1)
    return idxy


def draw_transmored_xy(image, f, *, width=1):
    h, w = image.shape[:2]
    
    T = draw_transform_matrix(image)
    num = int(np.ceil(max_distance(image))) * 4 # small change of y will lead to gap, use extra multiplier to cancel
    # x = np.linspace(0, 1, num)
    x = np.linspace(-1, 1, num)
    y = f(x)
    
    Txy = T @ np.stack([x, y, np.ones_like(x)]) # (3 x 3), (3, n) -> (3, n) 
    idxy = np.floor(Txy).astype(np.int)
    idxy = idxy[:2, :]
    idxy = box_idxy(image, idxy)
    
    if width == 1:
        return idxy
    idxy_list = [idxy]

    for ww in range(2, width+1):
        r = ww // 2
        if ww % 2 == 0:
            pos = np.stack([idxy[0, :], idxy[1, :] + r])
            idxy_list.append(pos)
        else:
            neg = np.stack([idxy[0, :], idxy[1, :] - r])
            idxy_list.append(neg)
    return box_idxy(image, np.concatenate(idxy_list, axis=1))


def random_line(image=None):
    if image is None:
        image = deform(image)
        
    h, w = image.shape[:2]
    
    line_color = sample_rgb_dist(text_rgb_free_dist)
    
    for i in range(8):
        k = np.random.random() * 10
        a = np.random.random() * 1 + 5
        width = np.random.randint(1, 5)
        idxy = draw_transmored_xy(image, lambda x: a*np.sin(k*x), width=width)
        # idxy = draw_transmored_xy(image, lambda x: np.zeros_like(x))
        image[idxy[1, :], idxy[0, :]] = line_color
    
    return image


def random_noise_points(image=None):
    if image is None:
        image = random_line(image)
    
    n = 100
    xv = np.random.randint(1, image.shape[1]-1, n)
    yv = np.random.randint(1, image.shape[0]-1, n)
    d = np.random.randint(-1, 2, (2, n, 3)) 
    d2 = np.cumsum(d, axis=2)
    xv2 = xv.reshape(n, 1) + d2[0]
    yv2 = yv.reshape(n, 1) + d2[1]
    
    idxy = box_idxy(image, np.stack([xv2.ravel(), yv2.ravel()]))
    noise_color = sample_rgb_dist(text_rgb_free_dist)
    image[idxy[1, :], idxy[0, :]] = noise_color
    
    return image


def random_captcha():
    return random_noise_points()

def random_captcha_text(text, *, font=None, alpha=50, sigma=6):
    x = random_background()
    x = random_base(x, text=text, font=font)
    x = affine_transform(x)
    x = deform(x, alpha=alpha, sigma=sigma)
    x = random_line(x)
    x = random_noise_points(x)
    return x


class ImageCaptcha():

    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def generate_image(self, text):
        img = random_captcha_text(text)
        if img.shape[0] == self.height and img.shape[1] == self.height:
            return img
        return cv2.resize(img, (self.width, self.height))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ex_img_list = [cv2.cvtColor(cv2.imread("data/{:04}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(4)]

    def compare(f):
        for i in range(8):
            plt.subplot(4, 2, i+1)
            if i % 2 == 0:
                plt.imshow(ex_img_list[i // 2])
            else:
                plt.imshow(f())
        plt.show()


    compare(lambda: deform(alpha=50, sigma=6))
    compare(random_captcha)

    ic = ImageCaptcha(192, 64)
    compare(lambda: ic.generate_image("AB12"))
