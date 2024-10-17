#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/video.hpp>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

#include <cassert>

#include <sphericalsfm/spherical_estimator.h>
#include <sphericalsfm/so3.h>
#include <sphericalsfm/rotation_averaging.h>
#include <sphericalsfm/spherical_utils.h>

#include <RansacLib/ransac.h>

#include "spherical_sfm_tools.h"

using namespace sphericalsfm;

namespace sphericalsfmtools
{

    // from https://github.com/BAILOOL/ANMS-Codes/blob/master/C%2B%2B/QtProject/anms.h
    struct sort_pred
    {
        bool operator()(const std::pair<float, int> &left, const std::pair<float, int> &right)
        {
            return left.first > right.first;
        }
    };

    std::vector<cv::KeyPoint> brownANMS(std::vector<cv::KeyPoint> keyPoints, int numRetPoints)
    {
        std::vector<std::pair<float, int>> results;
        results.push_back(std::make_pair(FLT_MAX, 0));
        for (unsigned int i = 1; i < keyPoints.size(); ++i)
        { // for every keyPoint we get the min distance to the previously visited keyPoints
            float minDist = FLT_MAX;
            for (unsigned int j = 0; j < i; ++j)
            {
                float exp1 = (keyPoints[j].pt.x - keyPoints[i].pt.x);
                float exp2 = (keyPoints[j].pt.y - keyPoints[i].pt.y);
                float curDist = sqrt(exp1 * exp1 + exp2 * exp2);
                minDist = std::min(curDist, minDist);
            }
            results.push_back(std::make_pair(minDist, i));
        }
        std::sort(results.begin(), results.end(), sort_pred()); // sorting by radius
        std::vector<cv::KeyPoint> kp;
        for (int i = 0; i < numRetPoints; ++i)
            kp.push_back(keyPoints[results[i].second]); // extracting numRetPoints keyPoints

        return kp;
    }

    void adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint> &keypoints,
                                      const int numToKeep)
    {
        if (keypoints.size() < numToKeep)
        {
            return;
        }
        std::sort(keypoints.begin(), keypoints.end(),
                  [&](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs)
                  {
                      return lhs.response > rhs.response;
                  });

        std::vector<cv::KeyPoint> anmsPts;

        std::vector<double> radii;
        radii.resize(keypoints.size());
        std::vector<double> radiiSorted;
        radiiSorted.resize(keypoints.size());

        const float robustCoeff = 1.11;

        for (int i = 0; i < keypoints.size(); ++i)
        {
            const float response = keypoints[i].response * robustCoeff;
            double radius = std::numeric_limits<double>::max();
            for (int j = 0; j < i && keypoints[j].response > response; ++j)
            {
                radius = std::min(radius, cv::norm(keypoints[i].pt - keypoints[j].pt));
            }
            radii[i] = radius;
            radiiSorted[i] = radius;
        }

        std::sort(radiiSorted.begin(), radiiSorted.end(),
                  [&](const double &lhs, const double &rhs)
                  {
                      return lhs > rhs;
                  });

        const double decisionRadius = radiiSorted[numToKeep];
        for (int i = 0; i < radii.size(); ++i)
        {
            if (radii[i] >= decisionRadius)
            {
                anmsPts.push_back(keypoints[i]);
            }
        }

        anmsPts.swap(keypoints);
    }

    void drawmatches(const cv::Mat &image0, const cv::Mat &image1, const Features &features0, const Features &features1, const Matches &m01, cv::Mat &imageout)
    {
        cv::addWeighted(image0, 0.5, image1, 0.5, 0, imageout);
        for (Matches::const_iterator it = m01.begin(); it != m01.end(); it++)
        {
            size_t ind0 = it->first;
            size_t ind1 = it->second;

            cv::line(imageout, features0.points[it->first], features1.points[it->second], cv::Scalar(255, 255, 255));
        }
    }

    Matches filter_matches(const Features &features0, const Features &features1, const Matches &m01, float thresh)
    {
        // filter matches that are longer or shorter than the median
        // match vector by thresh or more

        // get vector lengths
        std::vector<float> lengths;
        for (Matches::const_iterator it = m01.begin(); it != m01.end(); it++)
        {
            size_t ind0 = it->first;
            size_t ind1 = it->second;

            cv::Point2f pt0 = features0.points[it->first];
            cv::Point2f pt1 = features1.points[it->second];
            float dx = pt0.x - pt1.x;
            float dy = pt0.y - pt1.y;
            float length = sqrtf(dx * dx + dy * dy);
            lengths.push_back(length);
        }

        // find median
        std::vector<float> lengths_copy = lengths;
        std::sort(lengths_copy.begin(), lengths_copy.end());
        int mid = lengths_copy.size() / 2;
        float median;
        if (lengths_copy.size() % 2 == 0)
            median = (lengths_copy[mid - 1] + lengths_copy[mid]) / 2;
        else
            median = lengths_copy[mid];

        int n = 0;
        Matches mout;
        for (Matches::const_iterator it = m01.begin(); it != m01.end(); it++)
        {
            if (fabsf(lengths[n++] - median) < thresh)
                mout[it->first] = it->second;
        }
        return mout;
    }

    class DetectorTracker
    {
    protected:
        double min_dist;   // minimum distance between existing points and detecting points
        const int xradius; // horizontal tracking radius
        const int yradius; // vertical tracking radius
        cv::Ptr<cv::SIFT> sift;

    public:
        DetectorTracker(double _min_dist = 0, double _xradius = 0, double _yradius = 0) : min_dist(_min_dist), xradius(_xradius), yradius(_yradius), sift(cv::SIFT::create(20000)) {}

        void detect(const cv::Mat &image, Features &features)
        {
            assert(image.type() == CV_8UC1);

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descs;

            int ntokeep = std::max(0, 4000 - (int)features.points.size());
            if (ntokeep == 0)
                return;
            if (features.empty())
            {
                sift->detect(image, keypoints);
            }
            else
            {
                cv::Mat mask(image.size(), CV_8UC1);
                mask = cv::Scalar(255);
                for (int i = 0; i < features.points.size(); i++)
                    cv::circle(mask, features.points[i], min_dist, cv::Scalar(0), -1);
                sift->detect(image, keypoints, mask);
            }

            adaptiveNonMaximalSuppresion(keypoints, ntokeep);
            std::cout << "after anms, we have " << keypoints.size() << " keypoints\n";
            sift->compute(image, keypoints, descs);

            for (int i = 0; i < keypoints.size(); i++)
            {
                features.points.push_back(keypoints[i].pt);
                features.descs.push_back(descs.row(i));
            }
        }

        void track(cv::Mat &image0, cv::Mat &image1,
                   const Features &features0, Features &features1,
                   Matches &m01)
        {
            const size_t npts0 = features0.size();

            const std::vector<cv::Point2f> &prevPts(features0.points);
            std::vector<cv::Point2f> nextPts;
            std::vector<uchar> status;
            std::vector<float> err;

            cv::calcOpticalFlowPyrLK(image0, image1, prevPts, nextPts, status, err, cv::Size(xradius * 2 + 1, yradius * 2 + 1));

            for (int i = 0; i < npts0; i++)
            {
                if (status[i] == 0)
                    continue;

                features1.points.push_back(nextPts[i]);
                features1.descs.push_back(features0.descs.row(i));
                m01[i] = features1.size() - 1;
            }
        }
    };

    void match(const Features &features0, const Features &features1, Matches &m01, double ratio = 0.75)
    {
        cv::BFMatcher matcher;

        std::vector<std::vector<cv::DMatch>> matches;

        // p1 = query, p0 = train
        matcher.knnMatch(features1.descs, features0.descs, matches, 2);

        for (int i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance < ratio * matches[i][1].distance)
            {
                m01[matches[i][0].trainIdx] = matches[i][0].queryIdx;
            }
        }
    }

    void match_points(const cv::Mat &descriptors0, const cv::Mat &descriptors1, Matches &m01, double ratio = 0.75)
    {
        cv::BFMatcher matcher;

        std::vector<std::vector<cv::DMatch>> matches;

        // p1 = query, p0 = train
        matcher.knnMatch(descriptors0, descriptors1, matches, 2);

        for (int i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance < ratio * matches[i][1].distance)
            {
                m01[matches[i][0].trainIdx] = matches[i][0].queryIdx;
            }
        }
    }

    void build_feature_tracks_realtime(const Intrinsics &intrinsics,
                                   std::vector<Keyframe> &keyframes,
                                   std::vector<ImageMatch> &image_matches,
                                   const double inlier_threshold,
                                   const double min_rot,
                                   const bool inward,
                                   const std::string &image_path_pattern,
                                   const int thres)
{
    Eigen::Matrix3d Kinv = intrinsics.getKinv();

    // 첫 번째 이미지 읽기
    cv::Mat image0_in, image0;
    image0_in = cv::imread(image_path_pattern); // 첫 번째 이미지 경로 설정
    if (image0_in.empty())
    {
        std::cout << "Error: Could not read the first image: " << image_path_pattern << "\n";
        exit(1);
    }

    // 이미지 색상 변환
    cv::cvtColor(image0_in, image0, image0_in.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_GRAY2BGR);

    int xradius = image0.cols / 48;
    int yradius = image0.rows / 48;
    double min_dist = 40;

    DetectorTracker detectortracker(min_dist, xradius, yradius);

    Features features0;
    // 첫 번째 프레임에서 특징 포인트 검출
    detectortracker.detect(image0, features0);
    keyframes.push_back(Keyframe(0, features0));
    image0.copyTo(keyframes.back().image);

    std::cout << "Detected " << features0.size() << " features in image 0\n";

    // 특징 포인트를 이미지에 표시하고 저장
    cv::Mat image_with_features0 = image0.clone(); // 이미지 복사
    std::vector<cv::KeyPoint> keypoints0;
    for (const auto &point : features0.points)
    {
        keypoints0.emplace_back(cv::KeyPoint(point.x, point.y, 1)); // cv::KeyPoint 객체 생성
    }

    if (!keypoints0.empty()) // keypoints0가 비어있지 않은 경우에만 처리
    {
        // drawKeypoints 호출 시 outImage에 원본 이미지 전달
        cv::drawKeypoints(image_with_features0, keypoints0, image_with_features0, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        
        // 이미지가 비어있지 않은지 확인
        if (!image_with_features0.empty())
        {
            cv::imwrite("/home/ircvlab/HYU-2024-Embedded/code/feature_points/frame_0_features.jpg", image_with_features0);
        }
        else
        {
            std::cerr << "Error: image_with_features0 is empty!" << std::endl;
        }
    }
    else
    {
        std::cerr << "Error: No keypoints detected in image 0!" << std::endl;
    }

    // 이후 코드 동일
    ransac_lib::LORansacOptions options;
    options.squared_inlier_threshold_ = inlier_threshold * inlier_threshold * Kinv(0) * Kinv(0);

    int kf_index = 1;             // Keyframe 인덱스
    int image_index = 1;          // 이미지 인덱스
    const int max_images = thres; // 처리할 최대 이미지 수
    cv::Mat image1_in, image1;

    // "frame_" 다음의 부분을 찾기
    std::size_t start_pos = image_path_pattern.find("frame_") + 6; // "frame_"의 길이는 6
    std::size_t end_pos = image_path_pattern.find(".jpg");

    // "000" 부분 파싱
    std::string number_str = image_path_pattern.substr(start_pos, end_pos - start_pos);

    // 문자열을 정수로 변환
    int first_frame_number = std::stoi(number_str);

    // 이미지 파일 처리 루프
    while (image_index < max_images) // 최대 이미지 수 체크
    {
        std::stringstream tt;
        tt << "/home/ircvlab/HYU-2024-Embedded/code/captured_frames/frame_"
           << std::setw(3) << std::setfill('0') << first_frame_number + image_index << ".jpg";

        std::string image_name = tt.str();
        std::cout << "Image name: " << image_name << std::endl;
        // 다음 이미지 경로 설정
        image1_in = cv::imread(image_name); // 다음 이미지 읽기

        if (image1_in.empty())
        {
            std::cout << "No more images to read." << std::endl;
            break; // 더 이상 읽을 이미지가 없으면 종료
        }

        cv::cvtColor(image1_in, image1, image1_in.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_GRAY2BGR);

        Features features1;
        Matches m01;

        // 현재 프레임에서 특징 포인트 검출
        detectortracker.detect(image1, features1);
        // 이전 프레임과 현재 프레임의 특징 매칭
        match(features0, features1, m01);
        std::cout << "Tracked " << features1.size() << " features in image " << image_index << "\n";

        std::vector<bool> inliers;
        int ninliers;

        // 상대 자세를 이용한 inlier 매칭 찾기
        RayPairList ray_pair_list;
        ray_pair_list.reserve(m01.size());
        for (const auto &match : m01)
        {
            size_t index0 = match.first;
            size_t index1 = match.second;

            Eigen::Vector3d loc0 = Eigen::Vector3d(features0.points[index0].x, features0.points[index0].y, 1);
            Eigen::Vector3d loc1 = Eigen::Vector3d(features1.points[index1].x, features1.points[index1].y, 1);
            loc0 = Kinv * loc0;
            loc1 = Kinv * loc1;

            Ray u, v;
            u.head(3) = loc0;
            v.head(3) = loc1;

            ray_pair_list.emplace_back(u, v);
        }

        // Essential matrix 추정
        SphericalEstimator estimator(ray_pair_list, inward);
        ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d, std::vector<Eigen::Matrix3d>, SphericalEstimator> ransac;
        Eigen::Matrix3d E;
        ransac_lib::RansacStatistics stats;
        ninliers = ransac.EstimateModel(options, estimator, &E, &stats);
        inliers.resize(ray_pair_list.size());
        for (int i = 0; i < ray_pair_list.size(); i++)
        {
            inliers[i] = (estimator.EvaluateModelOnPoint(E, i) < options.squared_inlier_threshold_);
        }

        fprintf(stdout, "%d: %lu matches and %d inliers (%0.2f%%)\n", image_index, m01.size(), ninliers, (double)ninliers / (double)m01.size() * 100);

        Eigen::Vector3d best_estimator_r;
        Eigen::Vector3d best_estimator_t;
        // Essential matrix에서 translation과 rotation 추출
        decompose_spherical_essential_matrix(E, inward, best_estimator_r, best_estimator_t);
        std::cout << best_estimator_r.transpose() << "\n";

        // 최소 회전 각도 확인
        if (best_estimator_r.norm() < min_rot * M_PI / 180.)
        {
            image_index++;
            continue;
        }

        Matches m01inliers;
        size_t i = 0;
        for (const auto &match : m01)
        {
            if (inliers[i++])
                m01inliers[match.first] = match.second;
        }

        // 프레임 업데이트
        image1.copyTo(image0);
        features0 = features1;

        std::cout << "features0 now has " << features0.size() << " features\n";
        // 새로운 포인트 검출 및 추가
        keyframes.push_back(Keyframe(image_index, features0));
        image0.copyTo(keyframes.back().image);

        // Keyframe 간 매칭된 feature point 저장
        image_matches.push_back(ImageMatch(kf_index - 1, kf_index, m01, so3exp(best_estimator_r), best_estimator_t));

        // 특징 포인트를 이미지에 표시하고 저장
        cv::Mat image_with_features1 = image1.clone(); // 이미지 복사
        std::vector<cv::KeyPoint> keypoints1;
        for (const auto &point : features1.points)
        {
            keypoints1.emplace_back(cv::KeyPoint(point.x, point.y, 1)); // cv::KeyPoint 객체 생성
        }

        if (!keypoints1.empty()) // keypoints1가 비어있지 않은 경우에만 처리
        {
            cv::drawKeypoints(image_with_features1, keypoints1, image_with_features1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
            
            // 이미지가 비어있지 않은지 확인
            if (!image_with_features1.empty())
            {
                cv::imwrite("/home/ircvlab/HYU-2024-Embedded/code/feature_points/frame_" + std::to_string(image_index) + "_features.jpg", image_with_features1);
            }
            else
            {
                std::cerr << "Error: image_with_features1 is empty!" << std::endl;
            }
        }
        else
        {
            std::cerr << "Error: No keypoints detected in image " << image_index << "!" << std::endl;
        }

        image_index++;
        kf_index++;
    }
}



    void build_feature_tracks(const Intrinsics &intrinsics,
                              std::vector<Keyframe> &keyframes,
                              std::vector<ImageMatch> &image_matches,
                              const double inlier_threshold,
                              const double min_rot,
                              const bool inward,
                              const std::string &video_path) // 비디오 경로 추가
    {
        Eigen::Matrix3d Kinv = intrinsics.getKinv();

        // OpenCV 비디오 캡처 객체 생성 (비디오 경로 사용)
        cv::VideoCapture cap(video_path); // 카메라 인덱스 0 대신 비디오 파일 경로 사용

        // 첫 번째 프레임 읽기
        cv::Mat image0_in, image0;
        if (!cap.read(image0_in))
        {
            std::cout << "Error: Could not read the first frame.\n";
            exit(1);
        }
        // 이미지 색상 변환
        if (image0_in.channels() == 3)
            cv::cvtColor(image0_in, image0, cv::COLOR_BGR2GRAY);
        else
            image0 = image0_in;

        int xradius = image0.cols / 48;
        int yradius = image0.rows / 48;

        std::cout << "image0 has size " << image0.cols << " " << image0.rows << "\n";
        std::cout << "radii: " << xradius << " " << yradius << "\n";
        double min_dist = 20;
        DetectorTracker detectortracker(min_dist, xradius, yradius);

        Features features0;
        // 첫 번째 프레임에서 특징 포인트 검출
        detectortracker.detect(image0, features0);
        keyframes.push_back(Keyframe(0, features0));
        image0.copyTo(keyframes.back().image);

        std::cout << "Detected " << features0.size() << " features in image 0\n";

        ransac_lib::LORansacOptions options;
        options.squared_inlier_threshold_ = inlier_threshold * inlier_threshold * Kinv(0) * Kinv(0);
        options.final_least_squares_ = true;
        ransac_lib::RansacStatistics stats;

        int kf_index = 1;    // Keyframe 인덱스
        int video_index = 1; // 비디오 프레임 인덱스
        cv::Mat image1_in, image1;

        // 비디오 프레임 처리 루프
        while (cap.read(image1_in))
        {
            if (image1_in.channels() == 3)
                cv::cvtColor(image1_in, image1, cv::COLOR_BGR2GRAY);
            else
                image1 = image1_in;

            Features features1;
            Matches m01;

            // 현재 프레임에서 특징 포인트 검출
            detectortracker.detect(image1, features1);
            // 이전 프레임과 현재 프레임의 특징 매칭
            match(features0, features1, m01);
            std::cout << "Tracked " << features1.size() << " features in image " << video_index << "\n";

            std::vector<bool> inliers;
            int ninliers;

            // 상대 자세를 이용한 inlier 매칭 찾기
            RayPairList ray_pair_list;
            ray_pair_list.reserve(m01.size());
            for (const auto &match : m01)
            {
                size_t index0 = match.first;
                size_t index1 = match.second;

                Eigen::Vector3d loc0 = Eigen::Vector3d(features0.points[index0].x, features0.points[index0].y, 1);
                Eigen::Vector3d loc1 = Eigen::Vector3d(features1.points[index1].x, features1.points[index1].y, 1);
                loc0 = Kinv * loc0;
                loc1 = Kinv * loc1;

                Ray u;
                u.head(3) = loc0;
                Ray v;
                v.head(3) = loc1;

                ray_pair_list.emplace_back(u, v);
            }

            // Essential matrix 추정
            SphericalEstimator estimator(ray_pair_list, inward);
            std::cout << "Running RANSAC on " << ray_pair_list.size() << " matches\n";

            ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d, std::vector<Eigen::Matrix3d>, SphericalEstimator> ransac;
            Eigen::Matrix3d E;
            ninliers = ransac.EstimateModel(options, estimator, &E, &stats);
            inliers.resize(ray_pair_list.size());
            for (int i = 0; i < ray_pair_list.size(); i++)
            {
                inliers[i] = (estimator.EvaluateModelOnPoint(E, i) < options.squared_inlier_threshold_);
            }

            fprintf(stdout, "%d: %lu matches and %d inliers (%0.2f%%)\n", video_index, m01.size(), ninliers, (double)ninliers / (double)m01.size() * 100);

            Eigen::Vector3d best_estimator_r;
            Eigen::Vector3d best_estimator_t;
            // Essential matrix에서 translation과 rotation 추출
            decompose_spherical_essential_matrix(E, inward, best_estimator_r, best_estimator_t);
            std::cout << best_estimator_r.transpose() << "\n";
            std::cout << best_estimator_t.transpose() << "\n";

            // 최소 회전 각도 확인
            if (best_estimator_r.norm() < min_rot * M_PI / 180.)
            {
                video_index++;
                continue;
            }

            Matches m01inliers;
            size_t i = 0;
            for (const auto &match : m01)
            {
                if (inliers[i++])
                    m01inliers[match.first] = match.second;
            }

            // 프레임 업데이트
            image1.copyTo(image0);
            features0 = features1;

            std::cout << "features0 now has " << features0.size() << " features\n";
            // 새로운 포인트 검출 및 추가
            keyframes.push_back(Keyframe(video_index, features0));
            image0.copyTo(keyframes.back().image);
            // Keyframe 간 매칭된 feature point 저장
            image_matches.push_back(ImageMatch(kf_index - 1, kf_index, m01, so3exp(best_estimator_r), best_estimator_t));

            video_index++;
            kf_index++;
        }

        // 비디오 캡처 종료
        cap.release();
    }

    int make_loop_closures(const Intrinsics &intrinsics, const std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches,
                           const double inlier_threshold, const int min_num_inliers, const int num_frames_begin, const int num_frames_end, const bool best_only, const bool inward)
    {
        int width = intrinsics.centerx * 2;
        int height = intrinsics.centery * 2;
        float thresh = 0.02 * sqrtf(width * width + height * height);
        Eigen::Matrix3d Kinv = intrinsics.getKinv();

        ransac_lib::LORansacOptions options;
        options.squared_inlier_threshold_ = inlier_threshold * inlier_threshold * Kinv(0) * Kinv(0);
        options.final_least_squares_ = true;
        ransac_lib::RansacStatistics stats;

        ImageMatch best_image_match(-1, -1, Matches(), Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        int loop_closure_count = 0;

        DetectorTracker detectortracker(0, 0, 0);

        int count0 = 0;
        for (int index0 = 0; index0 < keyframes.size(); index0++)
        {
            if (++count0 > num_frames_begin)
                break;

            const Features &features0 = keyframes[index0].features;

            int count1 = 0;
            // best loop closure 찾기 위해서 한 프레임이 있으면 뒤에 오는 모든 프레임에 대해서 추정함 - 이렇게 해서 가장 좋은 E를 주는 프레임들 선택
            for (int index1 = keyframes.size() - 1; index1 >= index0 + 1; index1--)
            {
                if (++count1 > num_frames_end)
                    break;

                const Features &features1 = keyframes[index1].features;

                Matches m01;

                match(features0, features1, m01);
                if (m01.size() < min_num_inliers)
                    continue;

                // cv::Mat matchesim;
                // drawmatches( keyframes[index0].image, keyframes[index1].image, features0, features1, m01, matchesim );
                // cv::imwrite( "matches" + std::to_string(keyframes[index0].index) + "-" + std::to_string(keyframes[index1].index) + ".png", matchesim );

                std::vector<bool> inliers;
                int ninliers;

                // find inlier matches using relative pose - feature point들을 가리키는 position vector인듯
                RayPairList ray_pair_list;
                ray_pair_list.reserve(m01.size());
                for (Matches::const_iterator it = m01.begin(); it != m01.end(); it++)
                {
                    cv::Point2f pt0 = features0.points[it->first];
                    cv::Point2f pt1 = features1.points[it->second];

                    Eigen::Vector3d loc0 = Eigen::Vector3d(pt0.x, pt0.y, 1);
                    Eigen::Vector3d loc1 = Eigen::Vector3d(pt1.x, pt1.y, 1);

                    loc0 = Kinv * loc0;
                    loc1 = Kinv * loc1;

                    Ray u;
                    u.head(3) = loc0;
                    Ray v;
                    v.head(3) = loc1;

                    ray_pair_list.push_back(std::make_pair(u, v));
                }

                SphericalEstimator estimator(ray_pair_list, inward);

                ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d, std::vector<Eigen::Matrix3d>, SphericalEstimator> ransac;
                ransac_lib::RansacStatistics stats;
                Eigen::Matrix3d E;
                if (m01.empty())
                {
                    ninliers = 0;
                }
                else
                {
                    // 여기서 essential matrix 반환 해줌
                    ninliers = ransac.EstimateModel(options, estimator, &E, &stats);
                    inliers.resize(ray_pair_list.size());
                    for (int i = 0; i < ray_pair_list.size(); i++)
                    {
                        inliers[i] = (estimator.EvaluateModelOnPoint(E, i) < options.squared_inlier_threshold_);
                    }
                }
                fprintf(stdout, "%d %d: %lu matches and %d inliers (%0.2f%%)\n",
                        keyframes[index0].index, keyframes[index1].index, m01.size(), ninliers, (double)ninliers / (double)m01.size() * 100);

                Matches m01inliers;
                size_t i = 0;
                for (Matches::const_iterator it = m01.begin(); it != m01.end(); it++)
                {
                    if (inliers[i++])
                        m01inliers[it->first] = it->second;
                }

                if (ninliers > min_num_inliers)
                {
                    Eigen::Vector3d best_estimator_t;
                    Eigen::Vector3d best_estimator_r;

                    // 이 과정으로 r과 t를 얻어냄
                    decompose_spherical_essential_matrix(E, inward, best_estimator_r, best_estimator_t);
                    std::cout << best_estimator_r.transpose() << "\n";
                    std::cout << best_estimator_t.transpose() << "\n";

                    // cv::Mat inliersim;
                    // drawmatches( keyframes[index0].image, keyframes[index1].image, features0, features1, m01inliers, inliersim );
                    // cv::imwrite( "inliers_before" + std::to_string(keyframes[index0].index) + "-" + std::to_string(keyframes[index1].index) + ".png", inliersim );

                    // m01inliers = filter_matches(features0,features1,m01inliers,thresh);

                    // drawmatches( keyframes[index0].image, keyframes[index1].image, features0, features1, m01inliers, inliersim );
                    // cv::imwrite( "inliers_after" + std::to_string(keyframes[index0].index) + "-" + std::to_string(keyframes[index1].index) + ".png", inliersim );

                    ImageMatch image_match(index0, index1, m01inliers, so3exp(best_estimator_r), best_estimator_t);
                    if (ninliers > best_image_match.matches.size())
                        best_image_match = image_match;
                    if (!best_only)
                    {
                        image_matches.push_back(image_match);
                        loop_closure_count++;
                    }
                }
            }
        }
        if (best_only)
        {
            if (best_image_match.matches.empty())
                return 0;
            image_matches.push_back(best_image_match);
            return 1;
        }
        return loop_closure_count;
    }

    void initialize_rotations(const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations)
    {
        rotations.resize(num_cameras);
        for (int i = 0; i < num_cameras; i++)
            rotations[i] = Eigen::Matrix3d::Identity();

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        rotations[0] = R;
        for (int index = 1; index < num_cameras; index++)
        {
            for (int i = 0; i < image_matches.size(); i++)
            {
                if (image_matches[i].index0 == index - 1 && image_matches[i].index1 == index)
                {
                    R = image_matches[i].R * R;
                    rotations[index] = R;
                    break;
                }
            }
        }
    }

    void initialize_positions(const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Vector3d> &positions, const std::vector<Eigen::Matrix3d> &rotations)
    {
        positions.resize(num_cameras);

        // 첫 번째 카메라의 위치를 원점으로 설정
        positions[0] = Eigen::Vector3d(0, 0, 0);

        // 두 번째 카메라부터 시작하여 위치 계산
        for (int index = 1; index < num_cameras; index++)
        {
            Eigen::Vector3d position = Eigen::Vector3d::Zero();

            for (const auto &match : image_matches)
            {
                // 현재 카메라와 이전 카메라의 인덱스를 비교
                if (match.index1 == index && match.index0 == index - 1)
                {
                    // 이전 카메라의 위치에 현재 카메라의 회전을 적용하고 translation 벡터를 더함
                    position = rotations[index] * positions[index - 1] + match.t; // match.t는 현재 프레임에서의 translation 벡터
                    break;
                }
            }

            // 각 카메라의 위치 업데이트
            positions[index] = position;
        }
    }

#include <Eigen/Dense>
#include <iostream>

    void compute_EulerAngles(const int num_cameras, const std::vector<Eigen::Matrix3d> &rotationMatrices, std::vector<Eigen::Vector3d> &euler_angles_vec)
    {
        // rotationMatrices의 크기 체크
        if (rotationMatrices.size() != num_cameras)
        {
            std::cerr << "Error: Number of cameras does not match the number of rotation matrices." << std::endl;
            return;
        }

        euler_angles_vec.resize(num_cameras); // num_cameras만큼 리사이즈

        for (int i = 0; i < num_cameras; ++i)
        {
            const Eigen::Matrix3d &rotationMatrix = rotationMatrices[i];

            // ZYX 순서로 오일러 각 계산
            Eigen::Vector3d euler_angle = rotationMatrix.eulerAngles(2, 1, 0); // ZYX 순서
            euler_angles_vec[i] = euler_angle;                                 // 계산한 오일러 각 저장
        }
    }

    void refine_rotations(const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations)
    {
        std::vector<RelativeRotation> relative_rotations;
        for (int i = 0; i < image_matches.size(); i++)
        {
            relative_rotations.push_back(RelativeRotation(image_matches[i].index0, image_matches[i].index1, image_matches[i].R));
        }

        optimize_rotations(rotations, relative_rotations);
    }

    // void build_sfm(std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches, const std::vector<Eigen::Matrix3d> &rotations,
    //                sphericalsfm::SfM &sfm, bool spherical, bool merge, bool inward)
    // {
    //     // 여러 프레임에서 같은 3d 포인트 추적 및 저장

    //     std::cout << "building tracks\n";
    //     for (int i = 0; i < keyframes.size(); i++)
    //     {
    //         keyframes[i].features.tracks.resize(keyframes[i].features.size());
    //         for (int j = 0; j < keyframes[i].features.size(); j++)
    //         {
    //             keyframes[i].features.tracks[j] = -1;
    //         }
    //     }

    //     // add cameras - 초기 위치를 첫 프레임으로부터 추정
    //     std::cout << "adding cameras\n";
    //     for (int index = 0; index < keyframes.size(); index++)
    //     {
    //         char path[1024];
    //         sprintf(path, "%06d.png", keyframes[index].index + 1);
    //         Eigen::Vector3d t(0, 0, -1);
    //         if (inward)
    //             t[2] = 1;
    //         int camera = sfm.AddCamera(Pose(t, so3ln(rotations[index])), path);
    //         sfm.SetRotationFixed(camera, (index == 0));
    //         sfm.SetTranslationFixed(camera, spherical ? true : (index == 0));
    //     }

    //     std::cout << "adding tracks\n";
    //     std::cout << "number of keyframes is " << keyframes.size() << "\n";
    //     for (int i = 0; i < image_matches.size(); i++)
    //     {
    //         const Matches &m01 = image_matches[i].matches;

    //         const int index0 = image_matches[i].index0;
    //         const int index1 = image_matches[i].index1;
    //         std::cout << "match between keyframes " << index0 << " and " << index1 << " has " << m01.size() << " matches\n";

    //         const Features &features0 = keyframes[index0].features;
    //         const Features &features1 = keyframes[index1].features;

    //         for (Matches::const_iterator it = m01.begin(); it != m01.end(); it++)
    //         {
    //             const cv::Point2f pt0 = features0.points[it->first];
    //             const cv::Point2f pt1 = features1.points[it->second];

    //             int &track0 = keyframes[index0].features.tracks[it->first];
    //             int &track1 = keyframes[index1].features.tracks[it->second];

    //             Observation obs0(pt0.x - sfm.GetIntrinsics().centerx, pt0.y - sfm.GetIntrinsics().centery);
    //             Observation obs1(pt1.x - sfm.GetIntrinsics().centerx, pt1.y - sfm.GetIntrinsics().centery);

    //             if (track0 != -1 && track1 == -1)
    //             {
    //                 track1 = track0;
    //                 sfm.AddObservation(index1, track0, obs1);
    //             }
    //             else if (track0 == -1 && track1 != -1)
    //             {
    //                 track0 = track1;
    //                 sfm.AddObservation(index0, track1, obs0);
    //             }
    //             else if (track0 == -1 && track1 == -1)
    //             {
    //                 track0 = track1 = sfm.AddPoint(Eigen::Vector3d::Zero(), features0.descs.row(it->first));

    //                 sfm.SetPointFixed(track0, false);

    //                 sfm.AddObservation(index0, track0, obs0);
    //                 sfm.AddObservation(index1, track1, obs1);
    //             }
    //             else if (track0 != -1 && track1 != -1 && (track0 != track1))
    //             {
    //                 if (merge)
    //                 {
    //                     // track1 will be removed
    //                     sfm.MergePoint(track0, track1);

    //                     // update all features with track1 and set to track0
    //                     for (int i = 0; i < keyframes.size(); i++)
    //                     {
    //                         for (int j = 0; j < keyframes[i].features.size(); j++)
    //                         {
    //                             if (keyframes[i].features.tracks[j] == track1)
    //                                 keyframes[i].features.tracks[j] = track0;
    //                         }
    //                     }
    //                 }
    //                 else
    //                 {
    //                     sfm.AddObservation(index0, track0, obs0);
    //                     sfm.AddObservation(index1, track1, obs1);
    //                 }
    //             }
    //         }
    //     }

    //     // std::cout << "retriangulating...\n";
    //     // sfm.Retriangulate();
    // }

    void show_reprojection_error(std::vector<Keyframe> &keyframes, SfM &sfm)
    {
        double focal = sfm.GetIntrinsics().focal;
        double centerx = sfm.GetIntrinsics().centerx;
        double centery = sfm.GetIntrinsics().centery;
        for (int i = 0; i < keyframes.size(); i++)
        {
            cv::Mat image;
            if (keyframes[i].image.channels() == 1)
            {
                cv::cvtColor(keyframes[i].image, image, cv::COLOR_GRAY2BGR);
            }
            else
            {
                keyframes[i].image.copyTo(image);
            }

            Pose pose = sfm.GetPose(i);

            // draw points and observations
            for (int point_index = 0; point_index < sfm.GetNumPoints(); point_index++)
            {
                // get observation
                Observation obs;
                if (!sfm.GetObservation(i, point_index, obs))
                    continue;

                // get point
                Point point = sfm.GetPoint(point_index);
                if (point.norm() == 0)
                    continue;

                Eigen::Vector3d proj3 = pose.apply(point);
                Eigen::Vector2d proj = focal * proj3.head(2) / proj3(2);

                // check re-proj error
                Eigen::Vector2d diff = proj - obs;
                double dist = diff.norm();

                dist = std::min(dist, 10.);
                cv::Scalar color = cv::Scalar(0, (1. - dist / 10.) * 255, 255);
                // if ( dist > 2. ) color = cv::Scalar(0,0,255);

                cv::Point2f loc(obs[0] + centerx, obs[1] + centery);
                cv::circle(image, loc, 3, color, -1);
            }

            char imagepath[1024];
            sprintf(imagepath, "reproj%06d.jpg", i);
            cv::imwrite(imagepath, image);
        }
    }
}
