import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#reading image
import pandas as pd
image_dir = '/home/bfirst/art_few_samples-20211120T141915Z-001'
# img1 = cv2.imread('/home/bfirst/Pulpit/1280px-Themistokles_von_Eckenbrecher_Utsikt_over_Lærdalsøren.jpeg')
img1 = cv2.imread('/home/bfirst/art_few_samples-20211120T141915Z-001/art_few_samples/marina-2.jpg')
# img1 = cv2.imread('/home/bfirst/art_few_samples-20211120T141915Z-001/art_few_samples/marina-3.jpg')[:, ::-1]
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
stride_axis = np.argmax(gray1.shape)
window_size = gray1.shape[1] if stride_axis == 0 else gray1.shape[0]
#keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)

keypoints_magnitudes = set((keypoint.pt[int(not stride_axis)], keypoint.response) for keypoint in keypoints_1)
window_scores = {'start': [], 'keypoints_quantity': [], 'keypoints_total_magnitude': []}
axis_size = gray1.shape[stride_axis]
right_boundary = axis_size - window_size
left_sided_windows = list(range(0, right_boundary, axis_size // 50))
right_sided_windows = list(range(right_boundary, 0, -axis_size // 50))

for windows in (left_sided_windows, right_sided_windows):
    for window_start in windows:
        window = (window_start, window_start + window_size)
        window_magnitudes = [keypoint_magnitude[1] for keypoint_magnitude in keypoints_magnitudes if window[1] >= keypoint_magnitude[0] >= window[0]]
        window_magnitudes_check = []
        window_scores['start'].append(window_start)
        window_scores['keypoints_quantity'].append(len(window_magnitudes))
        window_scores['keypoints_total_magnitude'].append(sum(np.array(window_magnitudes)))
scores = pd.DataFrame(window_scores)
scores.sort_values(by=['start'], inplace=True)

img_1 = cv2.drawKeypoints(gray1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
max_score_window_start = scores.start.loc[scores.keypoints_total_magnitude.idxmax()]
img_1 = cv2.rectangle(img_1, (max_score_window_start, 0), (max_score_window_start+window_size, window_size), (255,0,0), 2)
plt.imshow(img_1)

stop = 1