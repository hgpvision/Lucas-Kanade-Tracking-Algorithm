#pragma once
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>

#include<opencv2\opencv.hpp>
#include <cv.h>
#include <highgui.h>

class AccessData
{
public:
	int WriteData(std::string fileName, cv::Mat& matData);
	int LoadData(std::string fileName, cv::Mat& matData, int matRows = 0, int matCols = 0, int matChns = 0)
};