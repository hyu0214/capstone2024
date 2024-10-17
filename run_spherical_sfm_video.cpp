#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

#include "spherical_sfm_tools.h"
#include "spherical_sfm_io.h"
#include <sphericalsfm/so3.h>

using namespace sphericalsfm;
using namespace sphericalsfmtools;

DEFINE_string(intrinsics, "", "Path to intrinsics (focal centerx centery k1 k2 p1 p2 k3)");
DEFINE_string(video_path, "", "Path to input video file");
DEFINE_double(inlierthresh, 2.0, "Inlier threshold in pixels");
DEFINE_double(minrot, 1.0, "Minimum rotation between keyframes");
DEFINE_int32(mininliers, 100, "Minimum number of inliers to accept a loop closure");
DEFINE_int32(numbegin, 30, "Number of frames at beginning of sequence to use for loop closure");
DEFINE_int32(numend, 30, "Number of frames at end of sequence to use for loop closure");
DEFINE_bool(bestonly, false, "Accept only the best loop closure");
DEFINE_bool(noloopclosure, false, "Allow there to be no loop closures");
DEFINE_bool(inward, false, "Cameras are outward facing");

int main(int argc, char **argv) {
    srand(0);

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Load camera intrinsics
    double focal, centerx, centery;
    double k1, k2, p1, p2, k3; // 왜곡 계수 추가
    std::ifstream intrinsicsf(FLAGS_intrinsics);
    if (!intrinsicsf.is_open()) {
        std::cerr << "Error: Could not open intrinsics file: " << FLAGS_intrinsics << std::endl;
        return -1;
    }
    intrinsicsf >> focal >> centerx >> centery >> k1 >> k2 >> p1 >> p2 >> k3; // 왜곡 계수 읽기
    Intrinsics intrinsics(focal, centerx, centery);

    // 왜곡 계수 행렬 초기화
    cv::Mat distortion_coeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

    // Open video file using OpenCV
    cv::VideoCapture cap(FLAGS_video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << FLAGS_video_path << std::endl;
        return -1;
    }
    std::cout << "Video opened successfully." << std::endl;

    std::vector<Keyframe> keyframes;
    std::vector<ImageMatch> image_matches;
    std::vector<Eigen::Matrix3d> rotations;

    std::cout << "Starting processing video..." << std::endl;

    // Real-time loop for capturing frames and performing SfM
    cv::Mat frame;
    int frame_count = 0;

    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            std::cout << "End of video or empty frame." << std::endl;
            break;
        }

        // 왜곡 보정
        cv::Mat undistorted_frame;
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal, 0, centerx,
                                                           0, focal, centery,
                                                           0, 0, 1);
        cv::undistort(frame, undistorted_frame, camera_matrix, distortion_coeffs);

        // Process frame
        Keyframe keyframe;
        keyframe.image = undistorted_frame; // 왜곡 보정된 프레임 사용
        keyframe.index = frame_count;
        keyframes.push_back(keyframe);
        frame_count++;

        // Process every 30 frames for feature tracking
        if (keyframes.size() >= 30) {
            std::cout << "Tracking features in video stream..." << std::endl;
            build_feature_tracks(intrinsics, keyframes, image_matches, FLAGS_inlierthresh, FLAGS_minrot, FLAGS_inward, FLAGS_video_path);

            std::cout << "Initializing rotations..." << std::endl;
            initialize_rotations(keyframes.size(), image_matches, rotations);
            std::vector<Eigen::Vector3d> euler_angles;
            compute_EulerAngles(keyframes.size(), rotations, euler_angles);

            // 각 프레임 별 euler angle 출력
            for (int i = 0; i < keyframes.size(); ++i) {
                std::cout << "Frame " << keyframes[i].index << " Euler Angles: " 
                          << euler_angles[i].transpose() << std::endl; // transpose로 행을 출력
            }

            keyframes.clear();
            image_matches.clear();
            rotations.clear();
        }

        // Escape loop when 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            std::cout << "Exiting..." << std::endl;
            break;
        }
    }

    cap.release();
    return 0;
}
