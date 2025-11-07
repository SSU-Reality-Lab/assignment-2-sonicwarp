import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            GrayScale의 이미지가 입력으로 들어옴
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            각 픽셀에 대해서 해리스 코너 점수를 가지고 있는 배열인 HarrisImage를 출력해줌
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            각 픽셀에서의 그래디언트의 방향을 각도로 나타낸 orientationImage를 출력해줌
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        # shape는 (이미지의 세로높이(픽셀의 행 개수), 이미지의 가로너비(픽셀의 열 개수), 이미지의 색상 채널(RGB or GS))
        # 인덱스 2까지만 복사(높이, 너비)
        height, width = srcImage.shape[:2]

        # 해리스 이미지 배열 초기화
        harrisImage = np.zeros(srcImage.shape[:2])
        # 오리엔테이션 이미지 배열 초기화
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        
        # Sobel 필터를 이용하여, 각 픽셀에서의 밝기 변화량을 계산(기울기)
        Ix = ndimage.sobel(srcImage, axis=0) # 세로 방향 미분
        Iy = ndimage.sobel(srcImage, axis=1) # 가로 방향 미분
        # sigma = 0.5인 가우시안 필터를 적용
        w_p = ndimage.gaussian_filter(srcImage, sigma=0.5)
        
        # 각각을 제곱, 곱셈
        Ixx = Ix**2
        Ixy = Ix*Iy
        Iyy = Iy**2
        
        # Setting the standard deviation to 5 (more than 4 is acceptable)
        weighted_Ixx = ndimage.gaussian_filter(Ixx, sigma=0.5,truncate=5)
        weighted_Ixy = ndimage.gaussian_filter(Ixy, sigma=0.5, truncate=5)
        weighted_Iyy = ndimage.gaussian_filter(Iyy, sigma=0.5, truncate=5)

        det = weighted_Ixx * weighted_Iyy - weighted_Ixy**2
        trace = weighted_Ixx + weighted_Iyy

        harrisImage = det + .1*(trace**2)
        orientationImage = np.degrees(np.arctan2(Iy.flatten(), Ix.flatten()).reshape(orientationImage.shape))

        # TODO-BLOCK-END
        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood. (7x7 <- 비등방성으로 인함)
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        localMax = ndimage.maximum_filter(harrisImage, size=7)
        destImage = (harrisImage == localMax)
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.pt = (x, y)
                f.size = 10
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                # TODO-BLOCK-END
                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        # import pdb; pdb.set_trace()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        height, width = grayImage.shape[:2]
        print("desc Shape : " + str(desc.shape))

        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            d = np.zeros((5,5))
            for k in range(-2,3):
                for l in range(-2, 3):
                    if y+k >=0 and y+k < height and x+l >= 0 and x+l < width:
                        d[2+k, 2+l] = grayImage[y+k, x+l]
                        
            desc[i] = np.reshape(d, (25))
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            x, y = int(f.pt[0]), int(f.pt[1])
            angle = np.radians(f.angle)
            rotMx = np.array([[math.cos(angle), math.sin(angle), 0],
                    [-math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1]])
            
            scaleMx = np.eye(3)
            for k, s in enumerate([0.2, 0.2]):
                scaleMx[k, k] = s

            translateMx = np.eye(3)
            translateMx[0][2] = -x
            translateMx[1][2] = -y

            translateMx2 = np.array([[1, 0 , windowSize/2], [0, 1, windowSize/2]])

            trans1Mx = np.zeros((2, 3))
            trans2Mx = np.zeros((2, 3))
            trans3Mx = np.zeros((2, 3))
            trans1Mx = np.dot(rotMx, translateMx)
            # print (trans1Mx)

            trans2Mx = np.dot(scaleMx, translateMx)
            # print(trans2Mx)


            transMx = np.dot(translateMx2, np.dot(scaleMx, np.dot(rotMx, translateMx)))
          
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit 
            # variance. If the variance is negligibly small (which we 
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            variance = np.var(destImage)
            vector_zeroes=np.zeros(windowSize * windowSize)

            desc[i] = vector_zeroes if variance < 1e-10 else np.reshape((destImage - np.mean(destImage))/np.std(destImage), windowSize*windowSize)
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            # 짝지어진 커플 목록 :     
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image(첫 번째 이미지의 몇 번 특징점인지)
                    trainIdx: The index of the feature in the second image(두 번째 이미지의 몇 번 특징점인지)
                    distance: The distance between the two features(두 특징점이 얼마나 비슷한지)
        '''
        # 짝지어진 커플의 목록을 초기화(빈 배열)
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        
        # 첫번째 이미지와 두 번째 이미지의 특징점 간의 거리를 계산한 2차원 배열(유사도 측정, 유클리드 거리)
        # desc1의 i번째 특징점과 desc2의 j번째 특징점 간의 거리 = dist[i][j]
        dist = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
        
        # i를 키워가면서 desc1의 특징점들을 하나씩 살펴보고, desc2와의
        for i,l in enumerate(desc1):
            min_dist = np.argmin(dist[i]) # desc1의 i번째 특징점과 가장 유사도가 높았던 desc2의 j번째 특징점
            match = cv2.DMatch()
            match.queryIdx = i # 첫 번째 이미지의 i번 특징점
            match.trainIdx = int(min_dist) # 두 번째 이미지의 min_dist(가장 유사도가 높은(거리가 짧은))번 특징점
            match.distance = dist[i, int(min_dist)]
            matches.append(match)

        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        
        dist = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
        # 여기 까지 SSDFeatureMatcher와 동일

        for i,l in enumerate(desc1):
            sort_Idx = np.argsort(dist[i])
            SSD1 = float(dist[i, sort_Idx[0]])
            SSD2 = float(dist[i, sort_Idx[1]])

            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = int(sort_Idx[0])
            
            if SSD1 == 0:
                match.distance = 0
            else:
                match.distance = SSD1 / (SSD2 *1.0)

            matches.append(match)

        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

