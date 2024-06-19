
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/highgui/highgui_c.h"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl
using namespace cv;
using namespace std;
using namespace cv::detail;



vector<String> img_names;

//参数调整:
bool preview = false;
bool try_cuda = false;
//double work_megapix = 0.6;
double work_megapix = -1;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.3f;
#endif
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;

int main()
{

	// 如果参数是文件名，则拼接文件
	//如果参数是int，则打开摄像头
	VideoCapture cap1("./Lout2.mp4");
	VideoCapture cap2("./Rout2.mp4");

	double rate = 60;
	int delay = 1000 / rate;
	bool stop(false);
	Mat frame1;
	Mat frame2;
	Mat frame;
	int k = 100;

	namedWindow("cam1", CV_WINDOW_AUTOSIZE);
	namedWindow("cam2", CV_WINDOW_AUTOSIZE);
	namedWindow("stitch", CV_WINDOW_AUTOSIZE);

	if (cap1.isOpened() && cap2.isOpened())
	{
		cout << "*** ***" << endl;
		cout << "摄像头已启动！" << endl;
	}
	else
	{
		cout << "*** ***" << endl;
		cout << "警告：请检查摄像头是否安装好!" << endl;
		cout << "程序结束！" << endl << "*** ***" << endl;
		return -1;
	}

	cap1.set(CAP_PROP_FRAME_WIDTH, 1080);
	cap1.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap2.set(CAP_PROP_FRAME_WIDTH, 1080);
	cap2.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap1.set(CAP_PROP_FOCUS, 0);
	cap2.set(CAP_PROP_FOCUS, 0);

	//获取两幅图像，通过这两幅图像来估计摄像机参数
	while (k--)
	{
		if (cap1.read(frame1) && cap2.read(frame2))
		{
			imshow("cam1", frame1);
			imshow("cam2", frame2);
			imwrite("frame1.bmp", frame1);
			imwrite("frame2.bmp", frame2);
		}
	}

	//计算相机内参数及旋转矩阵等参数


#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif
#if 0
	cv::setBreakOnError(true);
#endif
	//读入图片
	img_names.push_back("frame1.bmp");
	img_names.push_back("frame2.bmp");

	// Check if have enough images
	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}
	double work_scale = 1, seam_scale = 1, compose_scale = 1; // compose_scale 调整分辨率

	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
	LOGLN("Finding features...");
#if ENABLE_LOG
	int64 t = getTickCount();
#endif
	Ptr<Feature2D> finder;
	if (features_type == "orb")
	{
		finder = ORB::create();
	}
	else if (features_type == "akaze")
	{
		finder = AKAZE::create();
	}
#ifdef HAVE_OPENCV_XFEATURES2D
	else if (features_type == "surf")
	{
		finder = xfeatures2d::SURF::create();
	}
#endif
	else if (features_type == "sift")
	{
		finder = SIFT::create();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}
	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;
	for (int i = 0; i < num_images; ++i)
	{
		full_img = imread(samples::findFile(img_names[i]));
		full_img_sizes[i] = full_img.size();
		if (full_img.empty())
		{
			LOGLN("Can't open image " << img_names[i]);
			return -1;
		}
		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}
		computeImageFeatures(finder, img, features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());
		resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
		images[i] = img.clone();
	}
	full_img.release();
	img.release();
	LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOG("Pairwise matching");
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher;
	if (matcher_type == "affine")
		matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
	else if (range_width == -1)
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
	else
		matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();
	LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	// Check if we should save matches graph
	if (save_graph)
	{
		LOGLN("Saving matches graph...");
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}
	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}
	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;
	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}
	Ptr<Estimator> estimator;
	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
		estimator = makePtr<HomographyBasedEstimator>();
	vector<CameraParams> cameras;
	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed.\n";
		return -1;
	}
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}
	Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
	else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
	else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return -1;
	}
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
		return -1;
	}
	// Find median focal length
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}
	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}
	LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<Point> corners(num_images);
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<UMat> masks(num_images);
	// Prepare images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}
	// Warp images and their masks
	Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
	if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarperGpu>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarperGpu>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarperGpu>();
	}
	else
#endif
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarper>();
		else if (warp_type == "affine")
			warper_creator = makePtr<cv::AffineWarper>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarper>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarper>();
		else if (warp_type == "fisheye")
			warper_creator = makePtr<cv::FisheyeWarper>();
		else if (warp_type == "stereographic")
			warper_creator = makePtr<cv::StereographicWarper>();
		else if (warp_type == "compressedPlaneA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlaneA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniA2B1")
			warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniA1.5B1")
			warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniPortraitA2B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniPortraitA1.5B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "mercator")
			warper_creator = makePtr<cv::MercatorWarper>();
		else if (warp_type == "transverseMercator")
			warper_creator = makePtr<cv::TransverseMercatorWarper>();
	}
	if (!warper_creator)
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;
		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
		}
	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOGLN("Compensating exposure...");
#if ENABLE_LOG
	t = getTickCount();
#endif
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	if (dynamic_cast<GainCompensator*>(compensator.get()))
	{
		GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
		gcompensator->setNrFeeds(expos_comp_nr_feeds);
	}
	if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
	{
		ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
		ccompensator->setNrFeeds(expos_comp_nr_feeds);
	}
	if (dynamic_cast<BlocksCompensator*>(compensator.get()))
	{
		BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
		bcompensator->setNrFeeds(expos_comp_nr_feeds);
		bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
		bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
	}
	compensator->feed(corners, images_warped, masks_warped);
	LOGLN("Compensating exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOGLN("Finding seams...");
#if ENABLE_LOG
	t = getTickCount();
#endif
	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = makePtr<detail::NoSeamFinder>();
	else if (seam_find_type == "voronoi")
		seam_finder = makePtr<detail::VoronoiSeamFinder>();
	else if (seam_find_type == "gc_color")
	{

	//  寻找最佳拼接缝（求网络流的最小割）
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
	if (!seam_finder)
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
		return 1;
	}
	seam_finder->find(images_warped_f, corners, masks_warped);
	LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	// Release unused memory
	imwrite("mask.jpg",masks_warped);
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	///exposure&seam end///

	//实时拼接
	int i = 1; // 计数器，方便命名图片

	while (!stop)
	{
		if (cap1.read(frame1) && cap2.read(frame2))
		{
			imshow("cam1", frame1);
			imshow("cam2", frame2);
			imwrite("frame1.bmp", frame1);
			imwrite("frame2.bmp", frame2);

			//彩色帧转灰度
			//cvtColor(frame1, frame1, CV_RGB2GRAY);
			//cvtColor(frame2, frame2, CV_RGB2GRAY);


			//拼接过程
			//读入图片
			cout << "Compositing..." << endl;
#if ENABLE_LOG
			t = getTickCount();
#endif

			Mat img_warped, img_warped_s;
			Mat dilated_mask, seam_mask, mask, mask_warped;
			Ptr<Blender> blender;
			//double compose_seam_aspect = 1;
			double compose_work_aspect = 1;

			img_names.pop_back();
			img_names.pop_back();
			img_names.push_back("frame1.bmp");
			img_names.push_back("frame2.bmp");

			for (int img_idx = 0; img_idx < num_images; ++img_idx)
			{
				LOG("Compositing image #" << indices[img_idx] + 1);

				// Read image and resize it if necessary
				full_img = imread(img_names[img_idx]); // !!!!!!!!!!!!!!!!!!!!!!!!!!参数固定，可以试着读取不同图像
					if (!is_compose_scale_set)
					{
						if (compose_megapix > 0)
							compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
						is_compose_scale_set = true;

						// Compute relative scales
						//compose_seam_aspect = compose_scale / seam_scale;
						compose_work_aspect = compose_scale / work_scale;

						// Update warped image scale
						warped_image_scale *= static_cast<float>(compose_work_aspect);
						warper = warper_creator->create(warped_image_scale);

						// Update corners and sizes
						for (int i = 0; i < num_images; ++i)
						{
							// Update intrinsics
							cameras[i].focal *= compose_work_aspect;
							cameras[i].ppx *= compose_work_aspect;
							cameras[i].ppy *= compose_work_aspect;

							// Update corner and size
							Size sz = full_img_sizes[i];
							if (std::abs(compose_scale - 1) > 1e-1)
							{
								sz.width = cvRound(full_img_sizes[i].width * compose_scale);
								sz.height = cvRound(full_img_sizes[i].height * compose_scale);
							}

							Mat K;
							cameras[i].K().convertTo(K, CV_32F);
							Rect roi = warper->warpRoi(sz, K, cameras[i].R);
							corners[i] = roi.tl();
							sizes[i] = roi.size();
						}
					}
				if (abs(compose_scale - 1) > 1e-1)
					resize(full_img, img, Size(), compose_scale, compose_scale);
				else
					img = full_img;
				full_img.release();
				Size img_size = img.size();

				Mat K;
				cameras[img_idx].K().convertTo(K, CV_32F);

				// Warp the current image
				warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

				// Warp the current image mask
				mask.create(img_size, CV_8U);
				mask.setTo(Scalar::all(255));
				warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

				// Compensate exposure
				compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
				//imshow("seam_mask", img);
				img_warped.convertTo(img_warped_s, CV_16S);
				img_warped.release();
				img.release();
				mask.release();

				dilate(masks_warped[img_idx], dilated_mask, Mat());
				resize(dilated_mask, seam_mask, mask_warped.size());
				mask_warped = seam_mask & mask_warped;

				if (blender.empty())
				{
					blender = Blender::createDefault(blend_type, try_cuda);
					Size dst_sz = resultRoi(corners, sizes).size();
					float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
					if (blend_width < 1.f)
						blender = Blender::createDefault(Blender::NO, try_cuda);
					else if (blend_type == Blender::MULTI_BAND)
					{
						MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
						mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
						cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
					}
					else if (blend_type == Blender::FEATHER)
					{
						FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
						fb->setSharpness(1.f / blend_width);
						LOG("Feather blender, sharpness: " << fb->sharpness());
					}
					blender->prepare(corners, sizes);
				}

				// Blend the current image
				blender->feed(img_warped_s, mask_warped, corners[img_idx]);
			}
			Mat result, result_mask;
			blender->blend(result, result_mask);

			cout << "Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

			result.convertTo(frame, CV_8UC1);
			string outputImgName ="./out/"+ to_string(i) + ".jpg";
			imwrite(outputImgName,frame);
			imshow("stitch", frame);
			
			i++;
		}
		else
		{
			cout << "----------------------" << endl;
			cout << "waiting..." << endl;
		}

		if (waitKey(1) == 13)
		{
			stop = true;
			cout << "程序结束！" << endl;
			cout << "*** ***" << endl;
		}
	}
	return 0;
}
