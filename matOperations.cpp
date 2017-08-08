#pragma once
#include <opencv2\opencv.hpp>
#include "matOperations.h"
#include <math.h>

cv::Mat matOperations::seqMatRow(int rows, int cols)
{
	//**********************************************************
	//生成行号矩阵，即第一行全为0，第二行全为1，以此类推，返回CV_32S
	//输入：	rows：输出矩阵的行数
	//			cols：输出矩阵的列数
	//**********************************************************
	cv::Mat outM(rows, cols, CV_32S);
	for (int iter1=0;iter1 < rows;iter1++)
	{
		int* pRow = outM.ptr<int>(iter1);
		for (int iter2=0;iter2 < cols;iter2++)
		{
			pRow[iter2] = iter1;
		}
	}
	return outM;
}

cv::Mat matOperations::seqMatCol(int rows, int cols)
{
	//**********************************************************
	//生成列号矩阵，即第一列全为0，第二列全为1，以此类推，返回CV_32S
	//输入：	rows：输出矩阵的行数
	//			cols：输出矩阵的列数
	//**********************************************************

	cv::Mat outM(rows, cols, CV_32S);
	for (int iter1=0;iter1 < rows;iter1++)
	{
		int* pRow = outM.ptr<int>(iter1);
		for (int iter2=0;iter2 < cols;iter2++)
		{
			pRow[iter2] = iter2;
		}
	}
	return outM;
}

cv::Mat matOperations::gaussianKernelDeriX(double sigma)
{
	int kHsize = floor(3.0*sigma);
	int ksize = 2 * kHsize + 1;
	cv::Mat matX, matY;
	matX = seqMatCol(ksize, ksize) - kHsize;
	matY = seqMatRow(ksize, ksize) - kHsize;
	//std::cout << "The matX is: " << std::endl << matX << std::endl << std::endl;
	//std::cout << "The matY is: " << std::endl << matY << std::endl << std::endl;
	matX.convertTo(matX, CV_32F);
	matY.convertTo(matY, CV_32F);

	cv::Mat DGaussX(ksize, ksize, CV_32F);

	cv::Mat temp1, temp2, temp3;
	
	cv::divide(matX, 2.0 * PI*sigma*sigma*sigma*sigma, temp1);
	//std::cout << "The temp1 is: " << std::endl << temp1 << std::endl << std::endl;
	//std::cout << "The temp4is: " << std::endl << -matX.mul(matX) - matY.mul(matY) << std::endl << std::endl;
	
	cv::divide(-matX.mul(matX) - matY.mul(matY), 2.0 *sigma*sigma, temp2);
	//std::cout << "The temp2 is: " << std::endl << temp2 << std::endl << std::endl;
	cv::exp(temp2,temp3);
	//std::cout << "The temp3 is: " << std::endl << temp3 << std::endl << std::endl;
	DGaussX = -temp1.mul(temp3);
	//std::cout << "The DaussX is: " << std::endl << DGaussX << std::endl << std::endl;
	return DGaussX;
}

cv::Mat matOperations::gaussianKernelDeriY(double sigma)
{
	int kHsize = floor(3.0*sigma);
	int ksize = 2 * kHsize + 1;
	cv::Mat matX, matY;
	matX = seqMatCol(ksize, ksize) - kHsize;
	matY = seqMatRow(ksize, ksize) - kHsize;

	matX.convertTo(matX, CV_32F);
	matY.convertTo(matY, CV_32F);

	cv::Mat DGaussY(ksize, ksize, CV_32F);

	cv::Mat temp1, temp2, temp3;

	cv::divide(matY, 2.0 * PI*sigma*sigma*sigma*sigma, temp1);
	cv::divide(-matX.mul(matX) - matY.mul(matY), 2.0 *sigma*sigma, temp2);
	cv::exp(temp2, temp3);
	DGaussY = -temp1.mul(temp3);
	//std::cout << "The DaussY is: " << std::endl << DGaussY << std::endl << std::endl;
	return DGaussY;
}