#pragma once
#include<opencv2\opencv.hpp>

#define PI 3.1415926

class matOperations
{
public:
	static cv::Mat seqMatRow(int rows, int cols);

	static cv::Mat seqMatCol(int rows, int cols);

	static cv::Mat gaussianKernelDeriX(double sigma);

	static cv::Mat gaussianKernelDeriY(double sigma);
private:

};