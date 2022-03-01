//
//  ORBTracker.h
//  ofxCv-ORB-tracker
//
//  Created by Pierre Thirion on 2021-10-12.
//

#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "ofxCv/Utilities.h"
#include "ofRectangle.h"

namespace ofxCv {

using namespace cv;
using namespace std;

class ORBTracker
{
    public:
        
        struct ORBTrackerData
        {
            
            int matches;
            int inliers;
            double ratio;
            int keypoints;
            double fps;

            ORBTrackerData() : matches(0),
                inliers(0),
                ratio(0),
                keypoints(0),
                fps(0.)
            {}

            ORBTrackerData& operator+=(const ORBTrackerData& op) {
                matches += op.matches;
                inliers += op.inliers;
                ratio += op.ratio;
                keypoints += op.keypoints;
                fps += op.fps;
                return *this;
            }
            ORBTrackerData& operator/=(int num)
            {
                matches /= num;
                inliers /= num;
                ratio /= num;
                keypoints /= num;
                fps /= num;
                return *this;
            }
        };
        ORBTrackerData stats;
    
//        ORBTracker() = default;
        ORBTracker(cv::Ptr<cv::Feature2D> _detector, cv::Ptr<cv::DescriptorMatcher> _matcher) :
            detector(_detector),
            matcher(_matcher)
        {}
         
        void setFirstFrame(const cv::Mat frame, ofRectangle roi/*, string title, Stats& stats*/);
        cv::Mat process(const cv::Mat frame, /*Stats& stats,*/ int id);
        cv::Ptr<cv::Feature2D> getDetector(){    return detector;    }
    
        std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints);
    
    
    
    
    protected:
        const double ransac_thresh = 2.5f; // RANSAC inlier threshold
        const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
        const int bb_min_inliers = 10; // Minimal number of inliers to draw bounding
        
    
        cv::Ptr<cv::Feature2D> detector;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        cv::Mat first_frame, first_desc;
        std::vector<cv::KeyPoint> first_kp;
        std::vector<cv::Point2f> object_bb;
    
        
};
}



