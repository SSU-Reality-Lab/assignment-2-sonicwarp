import os, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import features

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def save_heatmap(array, title, filename, cmap='jet'):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_keypoints(image, keypoints, filename):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0,255,0), -1)
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(filename, vis)

# -------------------------------------------------------------------
# 0️⃣ Load Images
# -------------------------------------------------------------------
img1 = cv2.imread('resources/yosemite1.jpg')
img2 = cv2.imread('resources/yosemite2.jpg')

img3 = cv2.imread('resources/yim01.jpg')
img4 = cv2.imread('resources/yim02.jpg')

gray1 = cv2.cvtColor(img1.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

gray3 = cv2.cvtColor(img3.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 1️⃣ Feature Computation (TODO1~6)
# -------------------------------------------------------------------
HKD = features.HarrisKeypointDetector()
SFD = features.SimpleFeatureDescriptor()
MFD = features.MOPSFeatureDescriptor()

# TODO1
a1, b1 = HKD.computeHarrisValues(gray1)
a2, b2 = HKD.computeHarrisValues(gray2)

a3, b3 = HKD.computeHarrisValues(gray3)
a4, b4 = HKD.computeHarrisValues(gray4)

# TODO3
d1 = HKD.detectKeypoints(img1)
d2 = HKD.detectKeypoints(img2)

d3 = HKD.detectKeypoints(img3)
d4 = HKD.detectKeypoints(img4)

# Filter weak keypoints
d1 = [kp for kp in d1 if kp.response > 0.01]
d2 = [kp for kp in d2 if kp.response > 0.01]

d3 = [kp for kp in d3 if kp.response > 0.01]
d4 = [kp for kp in d4 if kp.response > 0.01]    

# TODO4~6
desc_simple_1 = SFD.describeFeatures(img1, d1)
desc_simple_2 = SFD.describeFeatures(img2, d2)
desc_mops_1 = MFD.describeFeatures(img1, d1)
desc_mops_2 = MFD.describeFeatures(img2, d2)

desc_simple_3 = SFD.describeFeatures(img3, d3)
desc_simple_4 = SFD.describeFeatures(img4, d4)
desc_mops_3 = MFD.describeFeatures(img3, d3)
desc_mops_4 = MFD.describeFeatures(img4, d4)

# -------------------------------------------------------------------
# 2️⃣ Visualization (TODO1, TODO3)
# -------------------------------------------------------------------
save_heatmap(a1, "Image1 - TODO1 Harris Strength", "results/img1_TODO1_harris_strength.png")
save_heatmap(a2, "Image2 - TODO1 Harris Strength", "results/img2_TODO1_harris_strength.png")

save_heatmap(a3, "Image3 - TODO1 Harris Strength", "results/img3_TODO1_harris_strength.png")
save_heatmap(a4, "Image4 - TODO1 Harris Strength", "results/img4_TODO1_harris_strength.png")

save_keypoints(img1, d1, "results/img1_TODO3_detected_keypoints.png")
save_keypoints(img2, d2, "results/img2_TODO3_detected_keypoints.png")
save_keypoints(img3, d3, "results/img3_TODO3_detected_keypoints.png")
save_keypoints(img4, d4, "results/img4_TODO3_detected_keypoints.png")

print("✅ Saved TODO1 & TODO3 visualizations.")

# -------------------------------------------------------------------
# 3️⃣ Matching (TODO7 - SSD, TODO8 - Ratio)
# -------------------------------------------------------------------
matcher_ssd = features.SSDFeatureMatcher()
matcher_ratio = features.RatioFeatureMatcher()

matcher_ssd_2 = features.SSDFeatureMatcher()
matcher_ratio_2 = features.RatioFeatureMatcher()    

# ------------------------------
# TODO7 - SSD matching
# ------------------------------
# Step 1. SSD matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ssd = matcher_ssd.matchFeatures(desc_mops_1,desc_mops_2)
matches_ssd_2 = matcher_ssd_2.matchFeatures(desc_mops_3,desc_mops_4)

# Step 2. 거리(distance)를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)[:150]
matches_ssd_2 = sorted(matches_ssd_2, key=lambda x: x.distance)[:150]  

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ssd_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ssd[:], None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches.png", ssd_vis)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches.png")

ssd_vis_2 = cv2.drawMatches(
    img3, d3, img4, d4, matches_ssd_2[:], None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches_2.png", ssd_vis_2)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches_2.png")    

# ------------------------------
# TODO8 - Ratio matching
# ------------------------------
# Step 1. Ratio matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ratio = matcher_ratio.matchFeatures(desc_mops_1, desc_mops_2)
matches_ratio_2 = matcher_ratio_2.matchFeatures(desc_mops_3, desc_mops_4)   

# Step 2. distance를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)[:150]
matches_ratio_2 = sorted(matches_ratio_2, key=lambda x: x.distance)[:150]   

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ratio_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ratio[:], None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO8_Ratio_matches.png", ratio_vis)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches.png")

ratio_vis_2 = cv2.drawMatches(
    img3, d3, img4, d4, matches_ratio_2[:], None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS   
)
cv2.imwrite("results/TODO8_Ratio_matches_2.png", ratio_vis_2)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches_2.png")    

print("🎯 All TODO1–8 visualizations done! Files saved in 'results/'")

'''
왜 TODO7의 SSD 매칭 결과보다 TODO8의 Ratio 매칭 결과가 더 우수한가요?
- SSDFeatureMathcher 경우
    img1의 각 특징점에 대해, img2에서 가장 거리가 가까운 특징점을 매칭시킵니다.
    dist[][] 배열에 desc1의 각 특징점과 desc2의 모든 특징점 간의 거리를 계산해서 넣어준 후, 가장 작은 값을 가진 특징점을 매칭시킵니다.
    min_dist = np.argmin(dist[i])를 통해 최근접 이웃을 찾는 방식입니다.
    그러나, 이 방식은 특징점을 거리로만 측정을 하므로, 매칭된 특징점이 실제로도 유사한 지에 대한 신뢰도를 제공하지 못합니다.
    
- RatioFeatureMatcher 경우
    img1의 각 특징점에 대해, img2에서 가장 가까운 특징점과 두 번째로 가까운 특징점 간의 거리 비율을 계산하여 매칭시킵니다.
    dist[][] 배열에 desc1의 각 특징점과 desc2의 모든 특징점 간의 거리를 계산해서 넣어준 후, 
    sort_Idx = np.argsort(dist[i])를 통해 가장 가까운 두 이웃의 인덱스를 찾습니다.
    if SSD1 == 0: ~~~ else: match.distance = SSD1 / ( SSD2 * 1.0 )을 통해 두 특징점 사이의 거리 자체가 아닌 거리 비율을 사용합니다.
    거리 비율을 사용하기 때문에, 매칭된 특징점이 실제로도 얼마나 유사한 지에 대한 신뢰도를 제공합니다.
    이 방법은 단순히 가장 가까운 이웃을 찾는 것보다 더 신뢰할 수 있는 매칭을 제공합니다.
'''