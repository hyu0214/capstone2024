#include <iostream>
#include <fstream>  // 파일 입출력을 위한 헤더 추가
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glob.h>
#include <cstdio>   // 파일 삭제를 위한 헤더 추가

#include "spherical_sfm_tools.h"
#include "spherical_sfm_io.h"
#include <sphericalsfm/so3.h>

using namespace sphericalsfm;
using namespace sphericalsfmtools;

// 명령줄 인자를 정의합니다.
DEFINE_string(intrinsics, "", "Path to intrinsics (focal centerx centery k1 k2 p1 p2 k3)");
DEFINE_string(image_path, "/home/ircvlab/HYU-2024-Embedded/code/captured_frames/frame_*.jpg", "Path to input image files");
DEFINE_double(inlierthresh, 2.0, "Inlier threshold in pixels");
DEFINE_double(minrot, 1.0, "Minimum rotation between keyframes");
DEFINE_int32(mininliers, 100, "Minimum number of inliers to accept a loop closure");
DEFINE_int32(numbegin, 30, "Number of frames at beginning of sequence to use for loop closure");
DEFINE_int32(numend, 30, "Number of frames at end of sequence to use for loop closure");
DEFINE_bool(bestonly, false, "Accept only the best loop closure");
DEFINE_bool(noloopclosure, false, "Allow there to be no loop closures");
DEFINE_bool(inward, false, "Cameras are outward facing");
DEFINE_int32(thres, 30, "Minimum number of frame");

int batch_index = 0;

int main(int argc, char **argv)
{
    srand(0);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Euler 각도 파일 삭제 후 새로 생성
    std::string euler_file_path = "/home/ircvlab/HYU-2024-Embedded/code/euler_angles.txt";
    std::remove(euler_file_path.c_str());  // 파일 삭제

    std::ofstream euler_file(euler_file_path);  // 새 파일 생성
    if (!euler_file.is_open()) {
        std::cerr << "Error: Could not open euler angles file." << std::endl;
        return -1;
    }

    // ZYX 순서 명시
    euler_file << "Euler angles calculated in ZYX (Yaw, Pitch, Roll) order.\n\n";

    // 카메라 내부 파라미터 로드
    double focal, centerx, centery;
    double k1, k2, p1, p2, k3;
    std::ifstream intrinsicsf(FLAGS_intrinsics);

    if (!intrinsicsf.is_open())
    {
        std::cerr << "Error: Could not open intrinsics file: " << FLAGS_intrinsics << std::endl;
        return -1;
    }
    intrinsicsf >> focal >> centerx >> centery >> k1 >> k2 >> p1 >> p2 >> k3;
    Intrinsics intrinsics(focal, centerx, centery);

    cv::Mat distortion_coeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

    std::vector<Keyframe> keyframes;
    std::vector<ImageMatch> image_matches;
    std::vector<Eigen::Matrix3d> rotations;

    std::cout << "Starting processing images..." << std::endl;

    glob_t glob_result;
    int glob_status = glob(FLAGS_image_path.c_str(), GLOB_TILDE, NULL, &glob_result);

    if (glob_status != 0)
    {
        std::cerr << "Error: Could not find any images at: " << FLAGS_image_path << std::endl;
        return -1;
    }

    cv::Mat frame;

    for (size_t frame_count = 0; frame_count < glob_result.gl_pathc; ++frame_count)
    {
        frame = cv::imread(glob_result.gl_pathv[frame_count]);
        if (frame.empty())
        {
            std::cerr << "Error: Could not open or find image: " << glob_result.gl_pathv[frame_count] << std::endl;
            continue;
        }

        cv::Mat undistorted_frame;
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal, 0, centerx,
                                 0, focal, centery,
                                 0, 0, 1);
        cv::undistort(frame, undistorted_frame, camera_matrix, distortion_coeffs);

        Keyframe keyframe;
        keyframe.image = undistorted_frame;
        keyframe.index = frame_count;
        keyframes.push_back(keyframe);

        if (keyframes.size() >= FLAGS_thres)
        {
            std::stringstream ss;
            ss << "/home/ircvlab/HYU-2024-Embedded/code/captured_frames/frame_"
               << std::setw(3) << std::setfill('0') << batch_index * FLAGS_thres << ".jpg";

            std::string image_path_pattern = ss.str();

            build_feature_tracks_realtime(intrinsics, keyframes, image_matches,
                                          FLAGS_inlierthresh, FLAGS_minrot,
                                          FLAGS_inward, image_path_pattern, FLAGS_thres);

            initialize_rotations(keyframes.size(), image_matches, rotations);

            std::vector<Eigen::Vector3d> euler_angles;
            compute_EulerAngles(keyframes.size(), rotations, euler_angles);

            // Euler 각도를 파일에 저장
            for (int i = 0; i < FLAGS_thres; ++i)
            {
                euler_file << "Frame " << batch_index * FLAGS_thres + i << " Euler Angles: "
                           << euler_angles[i].transpose() << std::endl;
            }

            keyframes.clear();
            image_matches.clear();
            rotations.clear();
            batch_index += 1;
        }
    }

    euler_file.close();  // 파일 닫기
    std::cout << "Euler angles saved to: " << euler_file_path << std::endl;
}
