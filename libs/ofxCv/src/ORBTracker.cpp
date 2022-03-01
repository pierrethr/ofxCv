//
//  ORBTracker.cpp
//  ofxCv-ORB-tracker
//
//  Created by Pierre Thirion on 2021-10-14.
//

#include "ofxCv/ORBTracker.h"

namespace ofxCv {
    using namespace cv;
    using namespace std;

    void ORBTracker::setFirstFrame(const cv::Mat frame, ofRectangle roi)
    {
        
//        Mat frame = toCv(cvimg.getPixels());
        vector<Point2f> bb;
        bb.push_back(Point2f(static_cast<float>(roi.getX()), static_cast<float>(roi.getY())));
        bb.push_back(Point2f(static_cast<float>(roi.getX()+roi.getWidth()), static_cast<float>(roi.getY())));
        bb.push_back(Point2f(static_cast<float>(roi.getX()+roi.getWidth()), static_cast<float>(roi.getY()+roi.getHeight())));
        bb.push_back(Point2f(static_cast<float>(roi.getX()), static_cast<float>(roi.getY()+roi.getHeight())));
        
        // create mask from bounding box
        cv::Point *ptMask = new cv::Point[bb.size()];
        const cv::Point* ptContain = { &ptMask[0] };
        int iSize = static_cast<int>(bb.size());
        for (size_t i=0; i<bb.size(); i++) {
            ptMask[i].x = static_cast<int>(bb[i].x);
            ptMask[i].y = static_cast<int>(bb[i].y);
        }
        first_frame = frame.clone(); // copy camera frame
        cv::Mat matMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
        detector->detectAndCompute(first_frame, matMask, first_kp, first_desc); // detect features
        stats.keypoints = (int)first_kp.size(); // store keypoints
    //    drawBoundingBox(first_frame, bb); // draw bounding box around first frame
    //    putText(first_frame, title, cv::Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
        object_bb = bb;
        delete[] ptMask;
        
    }

    cv::Mat ORBTracker::process(const cv::Mat frame, /*Stats& stats,*/ int id)
    {
        
        cv::TickMeter tm;
        std::vector<KeyPoint> kp;
        cv::Mat desc;

        tm.start();
        detector->detectAndCompute(frame, noArray(), kp, desc); // detect features on camera frame
        stats.keypoints = (int)kp.size(); // update num of keypoints

        std::vector< std::vector<DMatch> > matches;
        std::vector<KeyPoint> matched1, matched2;
        matcher->knnMatch(first_desc, desc, matches, 2); // detect matches
        for(unsigned i = 0; i < matches.size(); i++) {
            if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
                matched1.push_back(first_kp[matches[i][0].queryIdx]);
                matched2.push_back(      kp[matches[i][0].trainIdx]);
            }
        }
        stats.matches = (int)matched1.size();

        cv::Mat inlier_mask, homography;
        std::vector<cv::KeyPoint> inliers1, inliers2;
        std::vector<cv::DMatch> inlier_matches;
        if(matched1.size() >= 4) {
            homography = findHomography(Points(matched1), Points(matched2),
                                        RANSAC, ransac_thresh, inlier_mask);
        }
        tm.stop();
        stats.fps = 1. / tm.getTimeSec();

        // not enough matches, return
        if(matched1.size() < 4 || homography.empty()) {
            cv::Mat resMat;
            cv::hconcat(first_frame, frame, resMat);
            stats.inliers = 0;
            stats.ratio = 0;
            std::cout << "not enough for " << id << std::endl;
            
            return resMat;
        }
        
        // enough matches, continue
        for(unsigned i = 0; i < matched1.size(); i++) {
            if(inlier_mask.at<uchar>(i)) {
                int new_i = static_cast<int>(inliers1.size());
                inliers1.push_back(matched1[i]);
                inliers2.push_back(matched2[i]);
                inlier_matches.push_back(DMatch(new_i, new_i, 0));
            }
        }
        stats.inliers = (int)inliers1.size();
        stats.ratio = stats.inliers * 1.0 / stats.matches;

        std::vector<Point2f> new_bb;
        perspectiveTransform(object_bb, new_bb, homography);
        Mat frame_with_bb = frame.clone();
        /*
        if(stats.inliers >= bb_min_inliers) {
            drawGreenBoundingBox(frame_with_bb, new_bb); // draw bounding box of matches
        }
         */
        cv::Mat resMat;
        // draw matches and return Mat
        drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
                    inlier_matches, resMat,
                    Scalar(255, 0, 0), Scalar(255, 0, 0));
        
        cout << stats.matches << endl;
        return resMat;
    }

    std::vector<cv::Point2f> ORBTracker::Points(std::vector<cv::KeyPoint> keypoints)
    {
        std::vector<cv::Point2f> res;
        for(unsigned i = 0; i < keypoints.size(); i++) {
            res.push_back(keypoints[i].pt);
        }
        return res;
    }

}
