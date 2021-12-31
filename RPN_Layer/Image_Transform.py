from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

MyTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([400, 400])
]) # Defing PyTorch Transform

def Image_Prep(img):
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return  MyTransform(img)

if (__name__ == '__main__'):
    img = cv2.imread('Chess Pieces.v24-416x416_aug.coco/train/0b4ba28f0c759a11750a6430649b52e3_jpg.rf.79ce979766c7725eac584a892f2af5b1.jpg')

    plt.figure(figsize = (10,10))
    plt.imshow(img)
    plt.show()

    img = Image_Prep(img)

    print(img.shape)