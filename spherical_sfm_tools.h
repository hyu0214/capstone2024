#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <string>

#include <opencv2/core.hpp>

#include <sphericalsfm/sfm.h>

namespace sphericalsfmtools
{
    typedef std::pair<size_t, size_t> Match;
    typedef std::map<size_t, size_t> Matches;

    struct Features
    {
        std::vector<int> tracks;
        std::vector<cv::Point2f> points;
        cv::Mat descs;
        Features() : descs(0, 128, CV_32F) {}
        int size() const { return points.size(); }
        bool empty() const { return points.empty(); }
    };

    struct Keyframe
    {
        int index;
        Features features;
        cv::Mat image;

        // 기본 생성자
        Keyframe() : index(0), features(Features()), image(cv::Mat()) {}

        // 매개변수를 받는 생성자
        Keyframe(const int _index, const Features &_features) : index(_index), features(_features), image(cv::Mat()) {}
    };

    struct ImageMatch
    {
        int index0, index1;
        Matches matches;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        ImageMatch(const int _index0, const int _index1, const Matches &_matches, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_t) : index0(_index0), index1(_index1), matches(_matches), R(_R), t(_t) {}

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    void build_feature_tracks_realtime(const sphericalsfm::Intrinsics &intrinsics,
                                        std::vector<Keyframe> &keyframes,
                                        std::vector<ImageMatch> &image_matches,
                                        const double inlier_threshold,
                                        const double min_rot,
                                        const bool inward,
                                        const std::string &image_path_pattern, const int thres);

    void build_feature_tracks(const sphericalsfm::Intrinsics &intrinsics,
                              std::vector<Keyframe> &keyframes,
                              std::vector<ImageMatch> &image_matches,
                              const double inlier_threshold,
                              const double min_rot,
                              const bool inward = false,
                              const std::string &video_path = "");

    int
    make_loop_closures(const sphericalsfm::Intrinsics &intrinsics, const std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches,
                       const double inlier_threshold, const int min_num_inliers, const int num_frames_begin, const int num_frames_end, const bool best_only,
                       const bool inward = false);

    void initialize_rotations(const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations);
    void refine_rotations(const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations);

    void build_sfm(std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches, const std::vector<Eigen::Matrix3d> &rotations,
                   sphericalsfm::SfM &sfm, bool spherical = true, bool merge = true, bool inward = false);

    void show_reprojection_error(std::vector<Keyframe> &keyframes, sphericalsfm::SfM &sfm);

    void compute_EulerAngles(const int num_cameras, const std::vector<Eigen::Matrix3d> &rotationMatrices, std::vector<Eigen::Vector3d> &euler_angles_vec);
}