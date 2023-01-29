import os
import cv2
from cv2 import rotate
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import scipy
import matplotlib.image as mpimg

INPUT_PATH = "./THE1-Images/"
OUTPUT_PATH = "./Outputs/"

def read_image(img_path, rgb = True):
    if rgb:
        img = mpimg.imread(img_path)
    else:
        img = cv2.imread(img_path,0)
    return img

def write_image(img, output_path, rgb = True):
    plt.imsave(output_path,img)

def extract_save_histogram(img, path):
    plt.hist(img.ravel(),256,(0,256))
    plt.savefig(path,dpi=150)
    plt.show()


def rotate_image(img,  degree = 0, interpolation_type = "linear"):
    #interpolation type: "linear" or "cubic"
    #degree: 45 or 90
    if interpolation_type == "linear":
        img=scipy.ndimage.interpolation.rotate(img, degree, axes=(1, 0), reshape=False, output=None, mode='constant', cval=0.0, prefilter=False)
        bilinear_img = cv2.resize(img,None, fx = 0, fy = 0, interpolation = cv2.INTER_LINEAR)
        
    else:
        img=scipy.ndimage.interpolation.rotate(img, degree, axes=(1, 0), reshape=False, output=None, mode='constant', cval=0.0, prefilter=False)
        bilinear_img=rescale(img, 1, anti_aliasing=False, multichannel=True, order = 3)

    return bilinear_img

def histogram_equalization(img):
    img_hist_eq=cv2.equalizeHist(img)
    print(img_hist_eq.shape)
    return img_hist_eq

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    #PART1
    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a1_45_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a1_45_cubic.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "linear")
    write_image(output, OUTPUT_PATH + "a1_90_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "cubic")
    write_image(output, OUTPUT_PATH + "a1_90_cubic.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a2_45_linear.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a2_45_cubic.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 90, "linear")
    write_image(output, OUTPUT_PATH + "a2_90_linear.png")

    #PART2
    img = read_image(INPUT_PATH + "b1.png", rgb = False)
    extract_save_histogram(img, OUTPUT_PATH + "original_histogram.png")
    equalized = histogram_equalization(img)
    extract_save_histogram(equalized, OUTPUT_PATH + "equalized_histogram.png")
    write_image(output, OUTPUT_PATH + "enhanced_image.png")

    # BONUS
    # Define the following function
    # equalized = adaptive_histogram_equalization(img)
    # extract_save_histogram(equalized, OUTPUT_PATH + "adaptive_equalized_histogram.png")
    # write_image(output, OUTPUT_PATH + "adaptive_enhanced_image.png")
