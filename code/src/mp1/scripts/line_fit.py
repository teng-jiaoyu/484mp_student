import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.signal import find_peaks

# feel free to adjust the parameters in the code if necessary


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, binary_warped.shape[1])
	plt.ylim(binary_warped.shape[0], 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, Minv):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped > 0, binary_warped > 0, binary_warped > 0))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = np.polyval(left_fit, ploty)
	right_fitx = np.polyval(right_fit, ploty)
	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	result = cv2.warpPerspective(result, Minv, (binary_warped.shape[1], binary_warped.shape[0]))
	return result


def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = np.polyval(left_fit, ploty)
	right_fitx = np.polyval(right_fit, ploty)

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros_like(undist, dtype=np.uint8)

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result


class Line():
	def __init__(self, n):
		"""
		n is the window size of the moving average
		"""
		self.n = n
		self.detected = False

		# Polynomial coefficients: x = A*y^2 + B*y + C
		# Each of A, B, C is a "list-queue" with max length n
		self.A = []
		self.B = []
		self.C = []
		# Average of above
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		"""
		Gets most recent line fit coefficients and updates internal smoothed coefficients
		fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
		"""
		# Coefficient queue full?
		q_full = len(self.A) >= self.n

		# Append line fit coefficients
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		# Pop from index 0 if full
		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)

		# Simple average of line coefficients
		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)

		return (self.A_avg, self.B_avg, self.C_avg)


def lane_fit(binary_warped, nwindows=20, margin=50, minpix=10):
    """Track two independent lane traces with sliding windows and quadratic fits.

    Seeds come from the lowest band containing two peaks, allowing a shifted
    lane or a camera blind region. Failed/ambiguous pairs return None. No previous
    frame or assumed lane width is used to fabricate a missing boundary.
    """
    binary_warped = np.asarray(binary_warped)
    if binary_warped.ndim != 2 or not np.isfinite(binary_warped).all():
        raise ValueError("Expected a finite 2D lane mask")
    height, width = binary_warped.shape
    if (not isinstance(nwindows, (int, np.integer)) or nwindows < 1
            or height < nwindows or width < 2 or margin <= 0 or minpix < 1):
        raise ValueError("Invalid mask size or sliding-window parameters")
    binary = binary_warped > 0
    nonzeroy, nonzerox = np.nonzero(binary)
    if len(nonzerox) < 100:
        return None

    edges = np.linspace(height, 0, nwindows + 1, dtype=int)
    bands = list(zip(edges[1:], edges[:-1]))
    seeds = None
    # A short band avoids the false double peaks of a single diagonal boundary
    # accumulated over half of the whole image.
    for seed_window, (low, high) in enumerate(bands):
        histogram = binary[low:high].sum(axis=0).astype(float)
        histogram = np.convolve(histogram, np.ones(5) / 5, mode='same')
        peaks, _ = find_peaks(np.pad(histogram, (1, 1)),
                             height=max(2, .2 * (high - low)), distance=max(1, int(margin)))
        peaks = peaks - 1
        if len(peaks) >= 2:
            # Adjacent boundaries form a lane; choose the pair nearest the vehicle.
            pairs = list(zip(peaks[:-1], peaks[1:]))
            seeds = min(pairs, key=lambda pair: (abs(np.mean(pair) - width / 2),
                                                -min(histogram[pair[0]], histogram[pair[1]])))
            break
    if seeds is None:
        return None

    centres = np.asarray(seeds, dtype=float)
    momentum = np.zeros(2)
    indices = [[], []]
    for low, high in bands[seed_window:]:
        if centres[0] >= centres[1]:
            return None
        split = float(np.mean(centres))
        rows = (nonzeroy >= low) & (nonzeroy < high)
        for side in range(2):
            # Half-open row bands and an exclusive midpoint prevent shared pixels.
            own_side = nonzerox < split if side == 0 else nonzerox >= split
            good = np.flatnonzero(rows & own_side & (np.abs(nonzerox - centres[side]) <= margin))
            if len(good) >= minpix:
                new_centre = float(np.median(nonzerox[good]))
                momentum[side] = .6 * (new_centre - centres[side]) + .4 * momentum[side]
                centres[side] = new_centre
                indices[side].append(good)
            else:
                centres[side] += momentum[side]

    def fit_trace(groups):
        if not groups:
            return None
        selected = np.concatenate(groups)
        # Remove isolated segmentation outliers without filling gaps in the mask.
        for iteration in range(3):
            y, x = nonzeroy[selected], nonzerox[selected]
            if (len(selected) < 50 or np.unique(y).size < 3
                    or np.ptp(y) < .15 * height):
                return None
            try:
                fit = np.polynomial.Polynomial.fit(y, x, 2).convert().coef[::-1]
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return None
            if fit.shape != (3,) or not np.isfinite(fit).all():
                return None
            residual = x - np.polyval(fit, y)
            median = np.median(residual)
            threshold = max(3., 3 * 1.4826 * np.median(np.abs(residual - median)))
            keep = np.abs(residual - median) <= threshold
            if keep.all() or iteration == 2:
                return fit, selected
            selected = selected[keep]

    traces = [fit_trace(groups) for groups in indices]
    if any(trace is None for trace in traces):
        return None
    (left_fit, left_lane_inds), (right_fit, right_lane_inds) = traces
    overlap = min(nonzeroy[left_lane_inds].max(), nonzeroy[right_lane_inds].max()) - max(
        nonzeroy[left_lane_inds].min(), nonzeroy[right_lane_inds].min())
    if overlap < .15 * height:
        return None
    ploty = np.arange(height, dtype=float)
    left_fitx, right_fitx = np.polyval(left_fit, ploty), np.polyval(right_fit, ploty)
    # Include the reference row (height), where the error is anchored. A line
    # split in two or a crossing pair must not produce a centreline or an error.
    widths = np.polyval(right_fit - left_fit, np.arange(height + 1))
    if (not np.isfinite(widths).all() or widths.min() < margin
            or widths.max() > width or widths.max() > 3 * widths.min()):
        return None
    return {
        'left_fit': left_fit, 'right_fit': right_fit,
        'left_fitx': left_fitx, 'right_fitx': right_fitx, 'ploty': ploty,
        'nonzerox': nonzerox, 'nonzeroy': nonzeroy,
        'left_lane_inds': left_lane_inds, 'right_lane_inds': right_lane_inds,
        'out_img': np.dstack((binary, binary, binary)).astype(np.uint8) * 255,
    }

def perspective_transform(img, src, interpolation=cv2.INTER_LINEAR):
    """
    Get bird's eye view from input image
    """
    height, width = img.shape[:2]
    dst = np.float32([(0,0), (0,height), (width, height), (width, 0)])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)
    warped_img = cv2.warpPerspective(img, M, (width, height), flags=interpolation)
    return warped_img, M, Minv


def closest_point_on_polynomial(point, coeffs):
    """Global nearest point on the unrestricted curve x=P(y), in input units.

    Stationary distances satisfy (P(y)-x0)*P'(y)+(y-y0)=0. Evaluate all
    numerically real roots rather than choosing the first cubic root.
    """
    point = np.asarray(point, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    if (point.shape != (2,) or coeffs.ndim != 1 or not len(coeffs)
            or not np.isfinite(point).all() or not np.isfinite(coeffs).all()):
        raise ValueError("Expected a finite 2D point and finite polynomial coefficients")
    x0, y0 = point
    P = np.poly1d(coeffs)
    stationary = (P - x0) * P.deriv() + np.poly1d([1, -y0])
    roots = stationary.roots
    ys = roots.real[np.abs(roots.imag) <= 1e-7 * (1 + np.abs(roots.real))]
    candidates = np.column_stack((P(ys), ys))
    candidates = candidates[np.isfinite(candidates).all(axis=1)]
    if not len(candidates):
        raise ValueError("No finite real nearest-point candidate")
    distances = np.sum((candidates - point)**2, axis=1)
    return candidates[np.argmin(distances)]
