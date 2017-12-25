import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#=======================Start of Camera calibration==============================#
# read in all images and store in an array
images = glob.glob('camera_cal/calibration*.jpg')

# chessboard cols and rows number
cb_col,cb_row = 9,6
chessboard_dim = (cb_col,cb_row)

# get image size
image_cali = mpimg.imread(images[0])  
img_size = image_cali.shape[1::-1]

# arrays to store object points in 3D and image points in 2D
objpoints = [] # 3D points in real world space 
imgpoints = [] # 2D points in image plane

# prepare object points, like (0,0,0),(1,0,0)...
objp = np.zeros((cb_row*cb_col,3),np.float32)
objp[:,:2] = np.mgrid[0:cb_col,0:cb_row].T.reshape(-1,2) # x,y coordinates

# iterate over all images and fill in objpoints and imgpoints
for frame in images:
    # read in each image
    image = mpimg.imread(frame)
    
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dim)    
    # if corners are found, add object points and image points
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)
        
#use obtained object points and image points to calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
#===========================End of Camera calibration==============================#

#=======================Start of Thresholding==============================#
def sobelx_color_threshold(img, s_thresh=(170, 255), sx_thresh=(30, 100)):
    img = np.copy(img)
    
    # undistort image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)
#     l_channel = hls[:,:,1]
    # l_channel = img[:,:,0]
    s_channel = hls[:,:,2]
    
    # grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel and  x gradient
    s_binary = np.zeros_like(s_channel)
    s_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))| ((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
#     color_binary = np.dstack(( np.zeros_like(sxbinary), np.zeros_like(sxbinary), s_binary))
    
    return s_binary*255,undist
#===========================End of Thresholding==============================#

#=======================Start of Region of Interest==============================#
# image shape
imshape = image_cali.shape
# vertices for interest region mask
vertices = np.array([[(imshape[1]*0.1,imshape[0]-1),
                      (imshape[1]*0.45, imshape[0]*0.6), 
                      (imshape[1]*0.55, imshape[0]*0.6),
                      (imshape[1]*0.95,imshape[0]-1)]],
                    dtype=np.int32)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
#===========================End of Region of Interest==============================#

#=======================Start of Perspective Transform==============================#
# points in original image to project
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
# points projecting to destination image
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

def perspective_transform(img = None,src = None,dst = None):
    
    # use calibrated camera mtx, dist to undistort raw image
    warped, M, Minv = img,None,None

    # get M and Minv, the transform matrix and the inverse matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    # warp image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv
#===========================End of Perspective Transform==============================#


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # does it need calculate histogram to detect?
        self.cal_hist = True
        # was the line detected in previous N iterations?
        self.detected = []  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = 0.0     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.recent_fit = []
        #radius of curvature of the line in  km
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference of fit lane line X-coordinate between last and new fits
        self.diffs = 0.0

    def smooth(self, prev, curr, coeficient = 0.4):
        '''
         exponential smoothing
        :param prev: old value
        :param curr: new value
        :param coeficient: smoothing coef.
        :return:
        '''
        return curr*coeficient + prev*(1-coeficient)
        
# track for left line
left_line = Line()
# track for left line
right_line = Line()

# keep last n records
keep_max = 20

# Choose the number of sliding windows
nwindows = 9

# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

# max tolerant deviation from last average position
max_dev = 16

# threshold for restart
lower,higher = 0.2,0.8

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720000 # kilometers per pixel in y dimension
xm_per_pix = 3.7/700000 # kilometers per pixel in x dimension

# get nonzeros points
def get_nonzeros(binary_warped = None,
                 out_img = None, 
                 window_height = 0, 
                 leftx_current = 0,
                 rightx_current = 0,
                 window=0,
                 nonzerox=None,
                 nonzeroy=None,
                 margin = 100):
        # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 10) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 10) 
    
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                       (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    return good_left_inds,good_right_inds

def pipeline(image = None, left_line = left_line, right_line = right_line):
    
    sobelx_color,undist = sobelx_color_threshold(image)
    masked_edges = region_of_interest(sobelx_color,vertices)
    binary_warped, M, Minv = perspective_transform(masked_edges,src,dst)
        # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Set the width of the windows +/- margin
    margin = 100
    # arrays for detected left lane indexes and right lane indexes
    left_lane_inds, right_lane_inds = [],[]
    
    # If lane line not detected last time for either track, then restart blind search
    if left_line.cal_hist or right_line.cal_hist:
#         print('recalculate for left:{}, right:{}'.format(left_line.cal_hist,right_line.cal_hist))
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Step through the windows one by one
        for window in range(nwindows):
            good_left_inds,good_right_inds = get_nonzeros(binary_warped = binary_warped,
                                                            out_img = out_img,
                                                            window_height = window_height, 
                                                            leftx_current = leftx_current,
                                                            rightx_current = rightx_current,
                                                            window=window,
                                                            nonzerox=nonzerox,
                                                            nonzeroy=nonzeroy,
                                                            margin=margin)
            
            # try to find enough points for left lane line
            enough_left = len(good_left_inds) > minpix
            while(not enough_left):
                # increase searching window size
                margin += 10
                # try another time
                good_left_inds,_ = get_nonzeros(binary_warped = binary_warped, 
                                                            out_img=out_img,
                                                            window_height = window_height, 
                                                            leftx_current = leftx_current,
                                                            rightx_current = rightx_current,
                                                            window=window,
                                                            nonzerox=nonzerox,
                                                            nonzeroy=nonzeroy,
                                                            margin=margin)
                # if found or the window is too large then jump out and use the current found points
                if len(good_left_inds) > minpix or margin >= 150:
                    enough_left = True
                    
            # Append left indices to the list
            left_lane_inds.append(good_left_inds)  
            # resize window size 
            margin = 100
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            # try to find enough points for right lane line
            enough_right = len(good_right_inds) > minpix
            while(not enough_right):
                # increase searching window size
                margin += 10
                # try another time
                _,good_right_inds = get_nonzeros(binary_warped = binary_warped, 
                                                            out_img=out_img,
                                                            window_height = window_height, 
                                                            leftx_current = leftx_current,
                                                            rightx_current = rightx_current,
                                                            window=window,
                                                            nonzerox=nonzerox,
                                                            nonzeroy=nonzeroy,
                                                            margin=margin)
                # if found or the window is too large then jump out and use the current found points
                if len(good_right_inds) > minpix or margin >= 150:
                    enough_right = True
                    
            # Append right indices to the list
            right_lane_inds.append(good_right_inds)  
            # resize window size 
            margin = 100
                    
            if len(good_right_inds) > minpix:  
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
#         print('Left_line: {}, shape: {}'.format(left_line.best_fit,left_line.best_fit.shape))
        left_lane_inds = ((nonzerox > (left_line.best_fit[0]*(nonzeroy**2) + left_line.best_fit[1]*nonzeroy + 
                            left_line.best_fit[2] - margin)) & (nonzerox < (left_line.best_fit[0]*(nonzeroy**2) + 
                            left_line.best_fit[1]*nonzeroy + left_line.best_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_line.best_fit[0]*(nonzeroy**2) + right_line.best_fit[1]*nonzeroy + 
                            right_line.best_fit[2] - margin)) & (nonzerox < (right_line.best_fit[0]*(nonzeroy**2) + 
                            right_line.best_fit[1]*nonzeroy + right_line.best_fit[2] + margin)))  
        
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
     # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Evaluate detection
    left_use = ''
    # Calculate current left lane line base position
    left_base_P = np.absolute(left_fit[0]*(y_eval)**2+left_fit[1]*y_eval+left_fit[0])
    # not first detection
    if len(left_line.detected) > 1:
        # Check if fitted list deviation exceeds threshold
        if abs(left_base_P - left_line.bestx) > max_dev:
            # detection failed
            left_use = 'fails!'
            left_line.detected.append(False)
            # use last fitted coefficients
            left_fit = left_line.best_fit 
            # Recalculate current best X-coordinate and update    
            left_line.bestx = sum(left_line.recent_xfitted)/len(left_line.recent_xfitted)

        # or this detection is acceptable
        else: 
            #then update current tracked parameters
            # Add current detection result
            left_use = 'success!'
            left_line.detected.append(True)
            # smooth output 
            left_base_P = left_line.smooth(left_line.bestx, left_base_P)
            # add detection result to tracking list
            left_line.recent_xfitted.append(left_base_P)
            # check if exceeds max size
            if len(left_line.recent_xfitted) > keep_max:
                # Then delete least recent records
                del left_line.recent_xfitted[0]
            # smooth coefficients
            left_fit = left_line.smooth(left_line.best_fit, left_fit)
            # add current fit into tracking list
            left_line.recent_fit.append(left_fit)
            # check if exceeds max size
            if len(left_line.recent_fit) > keep_max:
                left_line.recent_fit = left_line.recent_fit[1:]
            # update current best fit
            left_line.best_fit = np.sum(left_line.recent_fit,axis = 0)/len(left_line.recent_fit)   
            left_line.bestx = left_base_P
#             left_fit = left_line.best_fit 
            
#             print('Check recent_fit: {}'.format(left_line.recent_fit))
#             print('Check result: {}'.format(left_line.best_fit))
    else:# First detection
        left_use = 'Start from initial!'
        left_line.bestx = left_base_P
        left_line.cal_hist = False
        left_line.detected.append(True)
        left_line.recent_xfitted.append(left_base_P)
#         print('Left fit is {} and type is: {}'.format(left_fit,type(left_fit)))
#         print('left_line is {} and type is: {}'.format(left_line.recent_fit[0],type(left_line.recent_fit[0])))
        left_line.recent_fit.append(left_fit)
        left_line.best_fit = left_fit
        
    # check if exceeds max size
    if len(left_line.detected) > keep_max:
        # Then delete least recent records
        del left_line.detected[0]
    
    left_detection_rates =1- sum(left_line.detected)/len(left_line.detected)
    # if bad detection exceeds 30%, then recalculate base positions in next frame    
    if left_detection_rates > lower and left_detection_rates < higher:
        left_line.cal_hist = True
        # update current best fit
#         left_line.best_fit = np.sum(left_line.recent_fit,axis = 0)/len(left_line.recent_fit)
#         print('Left lane deviated')
    elif left_detection_rates >= higher: # if failure rate exceeds 50%, then clear and recalculate
        # update current best fit
#         left_line.best_fit = np.sum(left_line.recent_fit,axis = 0)/len(left_line.recent_fit)
        left_line.cal_hist = True
        left_line.detected.clear()
        left_line.recent_xfitted.clear()
        left_line.bestx = 0
#         left_line.recent_fit =  []
#         left_line.best_fit = None
    else:# bad detection rates acceptable
        left_line.cal_hist = False

        
    # Get current right lane line base position
    right_base_P = np.absolute(right_fit[0]*(y_eval)**2+right_fit[1]*y_eval+right_fit[0])
    right_use = ''
        # not first detection
    if len(right_line.detected) > 1:
        # Check if fitted list deviation exceeds threshold
        if abs(right_base_P - right_line.bestx) > max_dev:
            # detection failed
#             print('Right lane not detected!')
            right_use = 'fails!'
            right_line.detected.append(False)
            # use best fitted coefficients
            right_fit = right_line.best_fit  
             # Recalculate current best X-coordinate and update    
            right_line.bestx = sum(right_line.recent_xfitted)/len(right_line.recent_xfitted)
        # or this detection is acceptable
        else: 
            #then update current tracked parameters
            right_use = "success!"
            # Add current detection result
            right_line.detected.append(True)
            # smooth output
            right_base_P = right_line.smooth(right_line.bestx, right_base_P)
            # add detection result to tracking list
            right_line.recent_xfitted.append(right_base_P)
            # check if exceeds max size
            if len(right_line.recent_xfitted) > keep_max:
                # Then delete least recent records
                del right_line.recent_xfitted[0]
           
            # smooth output
            right_fit = right_line.smooth(right_line.best_fit, right_fit)
            # add current fit into tracking list
            right_line.recent_fit.append(right_fit)
            # check if exceeds max size
            if len(right_line.recent_fit) > keep_max:
                right_line.recent_fit = right_line.recent_fit[1:]
            # update current best fit
            right_line.best_fit = np.sum(right_line.recent_fit,axis = 0)/len(right_line.recent_fit)   
            right_line.bestx = right_base_P
            # use best fitted coefficients
#             right_fit = right_line.best_fit  
    else:# First detection
        right_use = "Start from initial!"        
        right_line.bestx = right_base_P
        right_line.detected.append(True)
        right_line.recent_xfitted.append(right_base_P)
        right_line.recent_fit.append(right_fit)
        right_line.best_fit = right_fit
        
    # check if exceeds max size
    if len(right_line.detected) > keep_max:
        # Then delete least recent records
        del right_line.detected[0]
    
    right_detection_rates = 1-sum(right_line.detected)/len(right_line.detected)
    # if bad detection exceeds 30%, then recalculate base positions in next frame    
    if right_detection_rates > lower and right_detection_rates < higher:
        right_line.cal_hist = True
        # update current best fit
#         right_line.best_fit = np.sum(right_line.recent_fit,axis = 0)/len(right_line.recent_fit)
#         print('Left lane deviated')
    elif right_detection_rates >= higher: # if failure rate exceeds 50%, then clear and recalculate
        # update current best fit
#         right_line.best_fit = np.sum(right_line.recent_fit,axis = 0)/len(right_line.recent_fit)
        right_line.cal_hist = True
        right_line.detected.clear()
        right_line.recent_xfitted.clear()
        right_line.bestx = 0
        right_line.recent_fit =  []
        right_line.best_fit = None
    else:# bad detection rates acceptable
        right_line.cal_hist = False
    # End of evaluation
    
    # Don't know what to do with these two parameters ....
    left_line.line_base_pos = left_base_P
    right_line.line_base_pos = right_base_P
    
    # Begin of updating tracking class parameters
    left_line.diffs = np.absolute(left_base_P - left_line.bestx)*xm_per_pix      
    right_line.diffs = np.absolute(right_base_P - right_line.bestx)*xm_per_pix
    # End of updating tracking class parameters
    
    # Generate x values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
#     print('Left fit is {} and type is: {}'.format(left_fit,np.array(left_fit).shape))
#     print('Best fit is {} and type is: {}'.format(left_line.best_fit,np.array(left_line.best_fit).shape))
#     print('left_line is {} and type is: {}'.format(left_line.recent_fit[0],np.array(left_line.recent_fit[0]).shape))
    # Draw final lanes
      # base x-position of left lane line
    left_base_pos  = np.absolute(left_fit[0]*y_eval**2+left_fit[1]*y_eval+left_fit[2])
    # base x-position of left lane line
    right_base_pos = np.absolute(right_fit[0]*y_eval**2+right_fit[1]*y_eval+right_fit[2])

    
    # meters of deviation from mid-point meatured by detected lane lines
    deviation_from_mid_point = (img_size[0]/2-(left_base_pos+right_base_pos)/2)*xm_per_pix*1000
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # update tracking class parameters
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    cv2.putText(result, 'Left lane radius : {0:.3f} km'.format(left_curverad), (int(img_size[0]/4),30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
    cv2.putText(result, 'Right lane radius: {0:.3f} km'.format(right_curverad), (int(img_size[0]/4),60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
    cv2.putText(result, 'Deviation from line: {0:.1f} m'.format(deviation_from_mid_point), (int(img_size[0]/4),90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
    cv2.putText(result, 'Left lane diffs : {0:.3f}'.format(left_line.diffs), (int(img_size[0]/4),120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
    cv2.putText(result, 'Right lane diffs: {0:.3f}'.format(right_line.diffs), (int(img_size[0]/4),150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
#     cv2.putText(result, 'Right_fit : {}'.format(right_line.best_fit), (0,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
    # cv2.putText(result, 'Right lane {}: {}'.format(left_use,left_fit), (0,210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
    # cv2.putText(result, 'Left lane {}: {}'.format(right_use,right_fit), (0,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)


    
    return result

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os

# process video files
project_output = 'test_video/output_project_video.mp4'
clip1 = VideoFileClip("test_video/project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(os.path.join(project_output), audio=False,verbose = True, progress_bar = True)

challenge_output = 'test_video/output_challenge_video.mp4'
clip1 = VideoFileClip("test_video/challenge_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(os.path.join(challenge_output), audio=False,verbose = True, progress_bar = True)

harder_challenge_output = 'test_video/output_harder_challenge_video.mp4'
clip1 = VideoFileClip("test_video/harder_challenge_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(os.path.join(harder_challenge_output), audio=False,verbose = True, progress_bar = True)
