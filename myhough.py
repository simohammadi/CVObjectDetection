## ################################
#--------Image Cropping----#
###################################

import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

## Line_fitting
#  Used for cropping the image between spacers
def line_fitting(edge_image, sample):
	dst = edge_image
	cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
	cdstP = np.copy(cdst)


	lines = cv.HoughLines(dst, 1.2 ,np.pi / 180, 150, None, 0, 0)
	count_theta = 0
	count_rho = 0
	if lines is not None:
		for i in range(0, len(lines)):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
			if theta < 1 or rho <0:
				if theta < 1:
					cv.line(cdst, pt1, pt2, (0,0,255), 2, cv.LINE_AA)
					count_theta = count_theta + 1
					if count_theta  == 2:
						cv.line(cdst, pt1, pt2, (0,0,255), 2, cv.LINE_AA) == []
					count_theta = 1
				if rho < 0:
					cv.line(cdst, pt1, pt2, (0,0,255), 2, cv.LINE_AA)
					count_rho = count_rho + 1
					if count_rho  == 2:
						cv.line(cdst, pt1, pt2, (0,0,255), 2, cv.LINE_AA) == []
					count_rho = 1

			if count_theta == 1 and count_rho == 1:
				break

	points_intersection = list()
	for i in range(0,cdst.shape[1]):
		if cdst[1][i][0] == 0 and cdst[1][i][1] == 0 and cdst[1][i][2] == 255:
			points_intersection.append(i)


	img_crop = sample[10:cdst.shape[1],points_intersection[1]:points_intersection[3]]

	return img_crop


## Main
# For testing the line fitting
def main(argv):
	sample = cv.imread("/home/sina/Documents/abb/pictures/DSC_4468.jpg")
	edge_image = cv.Canny(sample, 50, 200, None, 3)

	img_crop = line_fitting(edge_image, sample)
	plt.imshow(img_crop)
	plt.show()


"""
    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    	
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.waitKey()
    return 0
"""
if __name__ == "__main__":
    main(sys.argv[1:])	

