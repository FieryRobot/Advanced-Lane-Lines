# Advanced Lane Line Detection
# for Udacity Self-Driving Nanodegree
#
# by Ed Voas, Copyright (c) 2017

import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip

import glob
import os
import pandas as pd
import shutil

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

ym_per_pix = 3 / 88  # meters per pixel in y dimension
xm_per_pix = 3.7 / 630  # meters per pixel in x dimension


def calibrateCamera(images, grid_size, image_size):
    objpoints = []
    imgpoints = []

    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

    for filename in images:
        img = mpimg.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print("Unable to find appropriate number of corners on " + filename)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    return mtx, dist


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        d = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        d = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_d = np.absolute(d)
    scaled = np.uint8(255 * abs_d / np.max(abs_d))

    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    scaled_sobel = np.uint8(255 * mag / np.max(mag))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_x = np.absolute(sobelx)
    abs_y = np.absolute(sobely)

    dir = np.arctan2(abs_y, abs_x)

    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return binary_output


def hls_channel_threshold(img, channel="h", thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        channel_data = hls[:, :, 0]
    elif channel == 'l':
        channel_data = hls[:, :, 1]
    else:
        channel_data = hls[:, :, 2]

    binary = np.zeros_like(channel_data)
    binary[(channel_data >= thresh[0]) & (channel_data <= thresh[1])] = 1

    return binary


def rgb_channel_threshold(img, channel="r", thresh=(0, 255)):
    if channel == 'r':
        channel_data = img[:, :, 0]
    elif channel == 'g':
        channel_data = img[:, :, 1]
    else:
        channel_data = img[:, :, 2]

    binary = np.zeros_like(channel_data)
    binary[(channel_data >= thresh[0]) & (channel_data <= thresh[1])] = 1

    return binary


def hsv_channel_threshold(img, channel="h", thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if channel == 'h':
        channel_data = hsv[:, :, 0]
    elif channel == 's':
        channel_data = hsv[:, :, 1]
    else:
        channel_data = hsv[:, :, 2]

    binary = np.zeros_like(channel_data)
    binary[(channel_data >= thresh[0]) & (channel_data <= thresh[1])] = 1

    return binary


def color_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    boundaries = [
        ([0, 53, 100], [80, 255, 255]),  # yellow
        # ([ 0, 23, 45], [ 64, 189, 255]),  # yellow
        ([23, 45, 184], [41, 255, 255]),
        #         ([ 0, 57, 90], [ 178, 209, 255]), # yellow
        ([0, 0, 208], [178, 255, 255])  # white
    ]

    result = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    for lower, upper in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors we like
        mask = cv2.inRange(hsv, lower, upper)

        # and add it to our final one
        result = cv2.bitwise_or(result, mask)
    return result


def color_mask2(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, np.uint8(lower), np.uint8(upper))


def yellow_mask(img):
    return cv2.inRange(img, np.array([150, 150, 0]), np.array([255, 255, 120]))


def enhance_lines(image):
    # mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 200))
    # sat_binary = hls_channel_threshold(image, 's', thresh=(170, 255))
    # light_binary = hls_channel_threshold(image, 'l', thresh=(190, 255))
    red_binary = rgb_channel_threshold(image, 'r', thresh=(200, 255))
    sobelx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(22, 124))
    sobely = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(22, 124))
    sobelmag = mag_thresh(image, sobel_kernel=3, mag_thresh=(20, 200))
    mask_binary = (color_mask(image) / 255).astype(np.uint8)
    # sobeldir = dir_threshold(image, sobel_kernel=3, thresh=(0.04, 0.45))

    combined = np.zeros_like(red_binary)
    combined[(((red_binary == 1) | (mask_binary == 1)) &
              ((sobelx == 1) | (sobely == 1) | (sobelmag == 1)))] = 1

    yellow_binary = yellow_mask(image)
    yellow_binary = (yellow_binary // 255).astype(np.uint8)
    another_yellow = color_mask2(image, [23, 45, 184], [41, 255, 255])
    another_yellow_binary = (another_yellow // 255).astype(np.uint8)

    # mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(20, 200))

    mask = color_mask(image)
    mask_binary = (mask // 255).astype(np.uint8)

    all_yellow = (yellow_binary == 1) | (another_yellow_binary == 1)
    # red_and_light = (red_binary == 1) & (light_binary == 1)

    # combination1 = (mask_binary == 1) | (light_binary == 1) | (yellow_binary == 1)
    # combination2 = (sobelx == 1) | (sobely == 1)  # | (sobeldir == 1)

    everything = np.zeros_like(red_binary)
    # everything[(combination1 & combination2) | all_yellow | red_and_light] = 255
    everything[all_yellow | (mask_binary == 1) | (red_binary == 1)] = 255

    return everything


class Line():

    def __init__(self, name):
        self.name = name
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.last_fitx = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.last_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.color = None
        self.last_fits = []
        self.last_der = None
        self.last_der1 = None
        self.last_der2 = None
        self.last_search_rectangles = []
        self.x_base = None

        # these two are for debugging/pipeline imaging
        self.last_inds = np.array([])
        self.last_rectangles = []
        self.fail_count = 0

    def new_update(self, nonzerox, nonzeroy, inds, rectangles=None):
        if rectangles is None:
            rectangles = []

        self.detected = False
        self.last_inds = np.array([])
        self.last_rectangles = rectangles

        if len(inds) >= 100:
            ploty = np.linspace(0, IMAGE_HEIGHT - 1, IMAGE_HEIGHT)

            new_fit = np.polyfit(nonzeroy[inds], nonzerox[inds], 2)
            new_fitx = poly_it(new_fit, ploty)

            # if self.name == 'left':
            #     if np.abs(new_fitx[719] - 390) > 50:
            #         print("I DETECT A LINE BASE THAT IS OFF THE MARK")

            # the top or bottom of the line is not allowed to go flying outward.
            # if self.bestx is not None:
            #     ycoords = nonzeroy[inds]
            #     xcoords = nonzeroy[inds]
            #     if np.abs(new_fitx[719] - self.bestx[719]) > 20:
            #         xcoords = np.append(nonzerox[inds], self.bestx[719])
            #         ycoords = np.append(nonzeroy[inds], 719)

            #     if np.abs(new_fitx[0] - self.bestx[0]) > 50:
            #         xcoords = np.append(nonzerox[inds], self.bestx[0])
            #         ycoords = np.append(nonzeroy[inds], 0)

            #         new_fit = find_best_poly(xcoords, ycoords)
            #         new_fitx = poly_it(new_fit, ploty)

            # get the derivative of the midpoint of the line
            new_deriv = 2 * new_fit[0] * 360 + new_fit[1]

            # if len(new_fit) == 3:
            #     new_deriv2 = 2 * new_fit[0] * 719 + new_fit[1]
            # else:
            #     new_deriv2 = new_fit[0]
            #     new_fit = np.insert(new_fit, 0, 0.0) #normalize it for the rest of this

            # i want the derivative at y=0, which ends up just being fit[1]
            # new_deriv = new_fit[1]

            # self.last_der1 = new_deriv1
            # self.last_der2 = new_deriv2

            # print("DERIV DIFF IS {}".format( np.abs(new_deriv1 - new_deriv2)))

            # right or wrong always save these (mostly to help debug)
            self.last_fit = new_fit
            self.last_fitx = new_fitx
            self.last_inds = inds

            # if (self.fit_stddev is not None) and np.abs(self.last_fit[0] - self.fit_mean[0]) > self.fit_stddev[0]:
            #     print("OUTSIDE STANDARD DEV")

            # if no valid deriv from last time or the last one and this one differ too much, we lose.
            if (self.last_der is not None) and np.abs(new_deriv - self.last_der) > 0.15:
                # print("LINE {} SEEMS TO HAVE CHANGED DIRECTION. old {}, new {}".format(self.name, self.last_der, new_deriv))
                pass
            else:
                self.detected = True
                self.last_der = new_deriv

                self.last_fits.append(new_fit)
                if len(self.last_fits) > 5:
                    self.last_fits.pop(0)

                self.best_fit = np.mean(np.array(self.last_fits), axis=0)
                self.bestx = poly_it(self.best_fit, ploty)
        # else:
        #     print("Not enough inds!!")

    def incrementFailCount(self):
        # if we have previous data, start to age it out
        if self.last_der is not None:
            self.fail_count += 1

            if self.fail_count > 20:
                # print("line failed too many times. resetting")
                self.last_der = None
                self.last_fits = []
                self.best_fit = None
                self.fail_count = 0


def poly_it(p, x):
    f = np.poly1d(p)
    return f(x)


class LineDetectionResults:

    def __init__(self, lane_inds=None, rectangles=None):
        if lane_inds is None:
            lane_inds = []
        if rectangles is None:
            rectangles = []
        self.lane_inds = lane_inds
        self.rectangles = rectangles


class LineDetector:

    def __init__(self, image, side='left'):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.midX = self.width // 2

        nonzero = image.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        # Choose the number of sliding windows
        self.nwindows = 9
        # Set height of windows
        self.window_height = np.int(self.height / self.nwindows)

        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50

    def slide_window(self, x_current):
        lane_inds = []
        results = LineDetectionResults()

        # Step through the windows one by one
        for window in range(self.nwindows):
            win_y_low = self.height - (window + 1) * self.window_height
            win_y_high = self.height - window * self.window_height
            win_x_low = x_current - self.margin
            win_x_high = x_current + self.margin

            # Save the windows for a visualization image
            results.rectangles.append([(win_x_low, win_y_low), (win_x_high, win_y_high)])

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((self.nonzeroy >= win_y_low) &
                         (self.nonzeroy < win_y_high) &
                         (self.nonzerox >= win_x_low) &
                         (self.nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.minpix:
                x_current = np.int(np.mean(self.nonzerox[good_inds]))

        results.lane_inds = np.concatenate(lane_inds)

        return results

    def inds_in_area(self, area):
        area_map = {}
        for i in range(len(area[0])):
            y_coord = area[0][i]
            x_coord = area[1][i]
            if y_coord in area_map:
                row_info = area_map[y_coord]
                if x_coord < row_info[0]:
                    row_info[0] = x_coord
                elif x_coord > row_info[1]:
                    row_info[1] = x_coord
            else:
                area_map[y_coord] = [x_coord, x_coord]

        inds = []
        for i in range(len(self.nonzerox)):
            x, y = self.nonzerox[i], self.nonzeroy[i]
            if y in area_map:
                row = area_map[y]
                if x >= row[0] and x <= row[1]:
                    inds.append(i)

        return inds

    def search_last_area(self, line):
        line_pts = get_search_area_poly(line.last_fitx, self.image.shape[0])
        line_area = get_search_area(self.image, line_pts)

        return self.inds_in_area(line_area)

    def detect(self, left_line, right_line):
        # Take a histogram of the bottom half of the image
        temp = np.uint8(self.image / 255)
        histogram = np.sum(temp[temp.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        self._detectOneLine(left_line, leftx_base)
        self._detectOneLine(right_line, rightx_base)

    def _detectOneLine(self, line, baseX, tryLastArea=False):
        if tryLastArea and line.last_fit is not None:
            inds = self.search_last_area(line)
            line.new_update(self.nonzerox, self.nonzeroy, inds)
        else:
            line.detected = False  # force it through the path below

        if not line.detected:
            if line.x_base is not None:
                baseX = line.x_base
            results = self.slide_window(baseX)

            line.new_update(self.nonzerox, self.nonzeroy, results.lane_inds, results.rectangles)
            if line.detected:
                line.x_base = baseX

        if not line.detected:
            line.incrementFailCount()


def find_lines(binary_warped, left_line, right_line):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    left_lane_inds = []
    right_lane_inds = []

    detector = LineDetector(binary_warped)

    detector.detect(left_line, right_line)

    for r in left_line.last_rectangles:
        cv2.rectangle(out_img, r[0], r[1], (0, 255, 0), 2)
    left_lane_inds = left_line.last_inds

    for r in right_line.last_rectangles:
        cv2.rectangle(out_img, r[0], r[1], (0, 255, 0), 2)
    right_lane_inds = right_line.last_inds

    if len(left_lane_inds) > 0:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    if len(right_lane_inds) > 0:
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img


def get_search_area_poly(fitx, image_height):
    margin = 50

    ploty = np.linspace(0, image_height - 1, image_height)

    line_window1 = np.array([np.transpose(np.vstack([fitx - margin, ploty]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + margin, ploty])))])
    line_pts = np.hstack((line_window1, line_window2))

    return line_pts


def get_search_area(image, poly):
    blank = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    cv2.fillPoly(blank, np.int_([poly]), 255)

    return blank.nonzero()


def get_search_area_img(image_shape, poly):
    blank = np.zeros((image_shape[0], image_shape[1]), dtype='uint8')
    cv2.fillPoly(blank, np.int_([poly]), 255)

    return blank


def overlay_lane(warped_binary, undist, left_fit, right_fit, MInv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    color_warp = np.zeros_like(undist).astype(np.uint8)

    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    if left_fit is not None:
        left_fitx = poly_it(left_fit, ploty)
    if right_fit is not None:
        right_fitx = poly_it(right_fit, ploty)

    # only draw the lane poly if we have both lines
    if left_fit is not None and right_fit is not None:
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    if left_fit is not None:
        points = np.stack((left_fitx, ploty), axis=1).astype(np.int32)
        cv2.polylines(color_warp, [points], False, (255, 255, 255), thickness=12)

    if right_fit is not None:
        points = np.stack((right_fitx, ploty), axis=1).astype(np.int32)
        cv2.polylines(color_warp, [points], False, (255, 255, 255), thickness=12)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = cv2.warpPerspective(color_warp, MInv, (warped_binary.shape[1], warped_binary.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)
    return result


def determine_curvature(line, ploty):
    y_eval = np.max(ploty)
    fitx = poly_it(line.best_fit, ploty)
    fit_scaled = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)
    radius = ((1 + (2 * fit_scaled[0] * y_eval * ym_per_pix + fit_scaled[1]) ** 2) ** 1.5) / np.absolute(2 * fit_scaled[0])
    return radius


def get_lane_position(image, leftLine, rightLine):
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    bottom_left = poly_it(leftLine.best_fit, imageHeight)
    bottom_right = poly_it(rightLine.best_fit, imageHeight)
    lane_center = int(bottom_left + (bottom_right - bottom_left) / 2.0)
    diff = int(imageWidth / 2.0) - lane_center
    return diff


def getCalibration():
    if not os.path.exists('calibration.pkl'):
        print("No saved calibration. Calibrating now...")
        images = glob.glob('camera_cal/calibration*.jpg')
        gridSize = (9, 6)
        mtx, dist = calibrateCamera(images, gridSize, (IMAGE_HEIGHT, IMAGE_WIDTH))
        print("Calibration complete")

        pickle.dump({'mtx': mtx, 'dist': dist}, open('calibration.pkl', 'wb'))
    else:
        print("Loading saved calibration...")
        with open('calibration.pkl', mode='rb') as f:
            calibration = pickle.load(f)
            mtx = calibration['mtx']
            dist = calibration['dist']
    return mtx, dist


def getWarpTransform():
    width, height = IMAGE_WIDTH, IMAGE_HEIGHT
    midpoint = width // 2

    # here we figure out what the points should be. I tried to leave
    # the hood of the car out and follow the lines as best I could
    # while using multipliers and not hard-coding any specific image
    # coordinates.
    bottom = height - int(height * 0.05)  # leave out the hood
    topOffset = int(width * .06)
    bottomOffset = int(width * .31)

    src = np.float32([(midpoint - bottomOffset, bottom),
                      (midpoint - topOffset, height - int(height * .35)),
                      (midpoint + topOffset, height - int(height * .35)),
                      (midpoint + bottomOffset, bottom)])

    margin = width / 4
    dst = np.float32([[margin, height],
                      [margin, 0],
                      [width - margin, 0],
                      [width - margin, height]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


# hook for the fl_image callback. Just call back into our global processor. I know, gross, right?
globalProcessor = None


def process_image(image):
    global globalProcessor
    return globalProcessor._processFrame(image)


class Processor:

    def __init__(self, filename, maskOnly, saveDebug):
        self.filename = filename
        self.outputMaskOnly = maskOnly
        self.saveDebugInfo = saveDebug
        self.frameNumber = 1
        self.leftLine = Line("left")
        self.rightLine = Line("right")
        self.show_debug_lane_info = False

        self.calibMtx, self.calibDist = getCalibration()
        self.warpMatrix, self.warpInverseMatrix = getWarpTransform()

    def process(self):
        suffix = self.filename.split('.')[1]
        if suffix == "mp4":
            self._processVideo()
        elif suffix == "jpg" or suffix == "jpeg":
            self._processImage()

    def _processVideo(self):
        filename = self.filename
        filename_parts = filename.split('.')

        if len(filename_parts) == 1:
            extension = "mp4"
        else:
            filename = filename_parts[0]
            extension = filename_parts[1]

        input_file = filename + '.' + extension
        output_file = filename + '-out.mp4'

        print("Processing video...")
        print("   mask_only: {}".format(self.outputMaskOnly))
        print("   full-debug: {}...".format(self.saveDebugInfo))

        # fl_image doesn't take a context, so... globals :-/
        global globalProcessor

        globalProcessor = self

        clip1 = VideoFileClip(input_file)  # .subclip(4, 6)
        white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(output_file, audio=False)

        globalProcessor = None

    def _processImage(self):
        print("Processing image " + self.filename + "...")
        image = mpimg.imread(self.filename)
        image = image[:, :, :3]  # get rid of any alpha channel

        result = self._processFrame(image)

        mpimg.imsave('saved.jpg', result)

    def _processFrame(self, image):
        if self.saveDebugInfo:
            mpimg.imsave("debug/images/frame" + str(self.frameNumber) + ".jpg", image)

        # correct distortion
        undist = cv2.undistort(image, self.calibMtx, self.calibDist, None, self.calibMtx)

        # sharpen a bit
        blur = cv2.GaussianBlur(undist, (0, 0), 3)
        sharpened = cv2.addWeighted(undist, 1.5, blur, -0.5, 0)

        # warp for top-down view
        warped = cv2.warpPerspective(sharpened, self.warpMatrix, (IMAGE_WIDTH, IMAGE_HEIGHT))
        enhanced = enhance_lines(warped)  # * 255

        if self.outputMaskOnly:
            return np.uint8(np.dstack((enhanced, enhanced, enhanced)))

        # mpimg.imsave("image.jpg", warped)

        # find the lines
        out_img = find_lines(enhanced, self.leftLine, self.rightLine)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

        # if self.leftLine.detected is False or self.rightLine.detected is False:
        #     print("frame {} failed! best_fit is {}".format(self.frameNumber, self.leftLine.best_fit))

        if self.saveDebugInfo:
            self._saveDebugInfo(out_img, ploty, enhanced.shape[0])

        # mpimg.imsave("outimg.jpg", out_img)
        # raise KeyboardInterrupt

        lane_position = "Lane Position: "
        lane_curvature = "Lane Curvature Radius: "

        if self.leftLine.best_fit is not None and self.rightLine.best_fit is not None:
            diff = get_lane_position(image, self.leftLine, self.rightLine)
            lane_position += "{:.2f}m".format(diff * xm_per_pix)

            # Define conversions in x and y from pixels space to meters
            ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
            left_curverad = determine_curvature(self.leftLine, ploty)
            right_curverad = determine_curvature(self.rightLine, ploty)

            lane_curvature += "left: {:.2f}m, right: {:.2f}m".format(left_curverad, right_curverad)
        else:
            lane_position += "unknown"
            lane_curvature += "unknown"

        # here we will draw what lines we have and color the lane if we have both
        result = overlay_lane(warped, undist, self.leftLine.best_fit, self.rightLine.best_fit, self.warpInverseMatrix)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, lane_position, (30, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(result, lane_curvature, (30, 80), font, 1, (255, 255, 255), 2)

        self.frameNumber += 1

        return result

    def _saveDebugInfo(self, out_img, ploty, height):
        font = cv2.FONT_HERSHEY_SIMPLEX

        points = np.stack((self.leftLine.last_fitx, ploty), axis=1).astype(np.int32)
        cv2.polylines(out_img, [points], False, (255, 255, 0), thickness=2)
        points = np.stack((self.rightLine.last_fitx, ploty), axis=1).astype(np.int32)
        cv2.polylines(out_img, [points], False, (255, 255, 0), thickness=2)

        if self.leftLine.best_fit is not None:
            points = np.stack((self.leftLine.bestx, ploty), axis=1).astype(np.int32)
            cv2.polylines(out_img, [points], False, (255, 165, 0), thickness=2)

        if self.rightLine.best_fit is not None:
            points = np.stack((self.rightLine.bestx, ploty), axis=1).astype(np.int32)
            cv2.polylines(out_img, [points], False, (255, 165, 0), thickness=2)

        search_area_overlay = np.zeros_like(out_img)

        # left_line_pts = get_search_area_poly(self.leftLine.last_fitx, height)
        # right_line_pts = get_search_area_poly(self.rightLine.last_fitx, height)

        # cv2.fillPoly(search_area_overlay, np.int_([left_line_pts]), (0, 255, 0))
        # cv2.fillPoly(search_area_overlay, np.int_([right_line_pts]), (0, 255, 0))

        diag_str = ""
        if self.leftLine.detected is False and self.rightLine.detected is False:
            diag_str = "both lines failed"
        elif self.leftLine.detected is False:
            diag_str = "left failed"
        elif self.rightLine.detected is False:
            diag_str = "right failed"

        if len(diag_str) > 0:
            cv2.putText(out_img, diag_str, (30, 40), font, 1, (255, 0, 0), 2)

        merged = cv2.addWeighted(out_img, 1.0, search_area_overlay, 0.3, 0)

        self._logToCSV()
        mpimg.imsave("debug/images/frame" + str(self.frameNumber) + "-detect.jpg", merged)

    def _logToCSV(self):
        d = {'left_fit_0': self.leftLine.last_fit[0],
             'left_fit_1': self.leftLine.last_fit[1],
             'left_fit_2': self.leftLine.last_fit[2],
             'left_deriv': self.leftLine.last_der,
             'left_ok': self.leftLine.detected,
             'right_fit_0': self.rightLine.last_fit[0],
             'right_fit_1': self.rightLine.last_fit[1],
             'right_fit_2': self.rightLine.last_fit[2],
             'right_deriv': self.rightLine.last_der,
             'right_ok': self.rightLine.detected,
             }

        df = pd.DataFrame(d, index=[self.frameNumber],
                          columns=['left_fit_0', 'left_fit_1', 'left_fit_2', 'left_deriv1', 'left_deriv2',
                                   'left_ok', 'right_fit_0', 'right_fit_1', 'right_fit_2', 'right_deriv1',
                                   'right_deriv2', 'right_ok'])

        writeHeader = not os.path.exists('debug/lines.csv')

        with open('debug/lines.csv', 'a') as f:
            df.to_csv(f, header=writeHeader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Do Line Detection.')
    parser.add_argument('file', metavar='file', type=str,
                        help='file path to process')
    parser.add_argument("--mask-only", help="only output the mask",
                        action="store_true", dest="maskOnly")
    parser.add_argument("--full-debug", help="output debug info for frames",
                        action="store_true", dest="fullDebug")

    args = parser.parse_args()
    if args.maskOnly:
        args.fullDebug = False

    if args.fullDebug:
        shutil.rmtree('debug', True)
        os.makedirs('debug/images', exist_ok=True)

    # Alright. Let's do this...
    p = Processor(args.file, args.maskOnly, args.fullDebug)
    p.process()
