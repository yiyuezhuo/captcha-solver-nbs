
import cv2
import numpy as np
from captcha_model import decode, device
from torchvision.transforms.functional import to_tensor
from pathlib import Path

def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img

def solve_tesseract(image):
    import pytesseract # lazy load
    return pytesseract.image_to_string(image, config='--psm 10')

def solve_ctc(model, img):
    # img: pil image or numpy.array
    img = cv2.resize(img, (192, 64))
    output = model(to_tensor(img).unsqueeze(0).to(device))
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    return decode(output_argmax[0])

def solve_ctc_raw(model, img):
    # img: pil image or numpy.array
    output = model(to_tensor(img).unsqueeze(0).to(device))
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    return decode(output_argmax[0])

def predict(model, image_list):
    model.eval()
    pred_list = [solve_ctc(model, img) for img in image_list]
    return pred_list

def predict_raw(model, image_list):
    model.eval()
    pred_list = [solve_ctc_raw(model, img) for img in image_list]
    return pred_list

def validate(pred_list, label_list):
    word_acc = np.mean(np.array(label_list) == np.array(pred_list))
    # char_acc = np.mean(np.array([list(s) for s in label_list]) == np.array([list(s) for s in pred_list]))
    true_char = 0
    false_char = 0
    for pred, label in zip(pred_list, label_list):
        if len(pred) != len(label):
            false_char += len(label)
        else:
            for c_pred, c_label in zip(pred, label):
                if c_pred == c_label:
                    true_char += 1
                else:
                    false_char += 1
    char_acc = true_char / (false_char + true_char)
    return word_acc, char_acc

def load_labeled_data(root=None):
    if root is None:
        root = Path(__file__).parent / "labeled_data"
    else:
        root = Path(root)
    
    img_list = []
    for i in range(50):
        path = root / "{:04}.jpg".format(i)
        img = load_image(str(path))
        img_list.append(img)

    with open(root / "labels.txt") as f:
        label_list = [line.strip() for line in f.readlines()]

    return img_list, label_list


if __name__ == "__main__":
    model = torch.load("ctc3.pth")
    img_list, label_list = load_labeled_data()
    pred_list = predict(model, image_list)
    word_acc, char_acc = validate(pred_list, label_list)
    print(f"word_acc:{word_acc}, char_acc:{char_acc}")
