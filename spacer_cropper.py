import cv2 as cv
from myhough import line_fitting

class spacer_cropper:

    cropped_img = None

    def __init__(self, fn):
        img = self.read_img(fn)
        seg_img = self.seg_img(img)
        self.cropped_img = self.crop_spacer(img, seg_img)
    
    def read_img(self, fn):
        return cv.imread(fn)
    
    def seg_img(self, img):
        return cv.Canny(img, 50, 200, None, 3)
    
    def crop_spacer(self, img, seg_img):
        return line_fitting(seg_img, img)
    
    def display_img(self, img, bound):
        cv.imshow("Displayed img" + str(bound), img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
def main():
    fn = "/home/sina/Documents/abb/transition_pictures/with_spacer/good_candidate/DSC_4762.jpg"
    sp = spacer_cropper(fn)
    sp.display_img(sp.cropped_img, "")

if __name__ == "__main__":
    main()
    