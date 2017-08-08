#pragma once
#include<opencv2\opencv.hpp>

class warpping
{
private:
	
public:

	static cv::Mat warpTransform(cv::Mat img, cv::Mat transMatrix, cv::Mat x, cv::Mat y);

	static cv::Mat floor(cv::Mat M);
	static cv::Mat ceil(cv::Mat M);

	static cv::Mat floor(cv::Mat M, int max, int min);
	static cv::Mat ceil(cv::Mat M, int max, int min);

};

