#pragma once
#include<opencv2\opencv.hpp>
#include "warpping.h"

cv::Mat warpping::warpTransform(cv::Mat img, cv::Mat transMatrix, cv::Mat x, cv::Mat y)
{
	//**********************************************************
	//本函数为仿射变换函数，输出的矩阵type为CV_32F
	//输入：	img：			为输入原始图像，type任意
	//			transMatrix：	为变换参数
	//			x：				为模板的列矩阵，对应img中的列号
	//			y：				为模板的行矩阵，对应img中的行号
	//**********************************************************

	if (img.depth() != 5)		img.convertTo(img, CV_32F);	//对输入矩阵进行类型转换，下面语句中为其定义的指针为float类型的
	int cols = img.cols;
	int rows = img.rows;

	cv::Mat Tlocalx, Tlocaly;		//
	//cv::Mat Tlocalx(rows_x, rows_x, CV_32F);
	//cv::Mat Tlocaly(rows_x, rows_x, CV_32F);

	float M11 = transMatrix.at<float>(0, 0);
	float M12 = transMatrix.at<float>(0, 1);
	float M13 = transMatrix.at<float>(0, 2);
	float M21 = transMatrix.at<float>(1, 0);
	float M22 = transMatrix.at<float>(1, 1);
	float M23 = transMatrix.at<float>(1, 2);

	cv::addWeighted(x, M11, y, M12, M13, Tlocalx);
	cv::addWeighted(x, M21, y, M22, M23, Tlocaly);
	
	//输入的M,x,y的元素可能是整数，如果事先没有强行指定M,x,y的类型，则opencv会默认其为CV_8U，
	//这样计算完之后的Tlocal就为CV_8U类型（会自动根据参与运算的参数决定类型）
	//最好将所有进入到这个函数的Mat统一为CV_32F，这样就没有这个问题了
	if (Tlocalx.depth() != 5)		Tlocalx.convertTo(Tlocalx, CV_32F);
	if (Tlocaly.depth() != 5)		Tlocaly.convertTo(Tlocaly, CV_32F);
		
	cv::Mat xBas0, xBas1, yBas0, yBas1;
	
	xBas0 = floor(Tlocalx, cols - 1, 0);	//这是对灰度坐标的向下取整，不是灰度值
	yBas0 = floor(Tlocaly, rows - 1, 0);

	int basex00, basey00, basex01, basey01, basex10, basey10, basex11, basey11;

	cv::Mat xCoe, yCoe;

	xBas0.convertTo(xBas0, CV_32F);		//注意，floor函数返回CV_32S，下面在subtract中要和CV_32F做减法，因此需提前转换，不然出错
	yBas0.convertTo(yBas0, CV_32F);

	cv::subtract(Tlocalx, xBas0, xCoe);
	cv::subtract(Tlocaly, yBas0, yCoe);

	//双线性插值系数（注意这里是点乘不是矩阵乘法）
	cv::Mat perc00 = (1.f - xCoe).mul(1.f - yCoe);		
	//cv::Mat perc1 = (1 - xCoe).mul(yCoe);
	//cv::Mat perc2 = xCoe.mul(1 - yCoe);
	cv::Mat perc01 = xCoe.mul(1.f - yCoe);
	cv::Mat perc10 = (1.f - xCoe).mul(yCoe);
	cv::Mat perc11 = xCoe.mul(yCoe);

	cv::Mat outM(x.size(), CV_32F);
	float* poutM;
	float intensity00, intensity01, intensity10, intensity11;
	float coe00, coe01, coe10, coe11;
	float* pxBas0;
	float* pyBas0;

	float* pImg1;
	float* pImg2;

	float* pPerc00;
	float* pPerc01;
	float* pPerc10;
	float* pPerc11;

	int windowRow = yBas0.rows;
	int windowCol = xBas0.rows;

	//经过仿射变换之后，窗口会有所旋转，不能直接用Range来去行列，而要根据变换之后窗口的每个元素中给的坐标到下帧图像中查找提取对应灰度值
	for (int iter1 = 0;iter1 < windowRow;iter1++)
	{
		pxBas0 = xBas0.ptr<float>(iter1);
		pyBas0 = yBas0.ptr<float>(iter1);

		pPerc00 = perc00.ptr<float>(iter1);
		pPerc01 = perc01.ptr<float>(iter1);
		pPerc10 = perc10.ptr<float>(iter1);
		pPerc11 = perc11.ptr<float>(iter1);

		poutM = outM.ptr<float>(iter1);
		for (int iter2 = 0;iter2 < windowCol;iter2++)
		{
			basex00 = pxBas0[iter2];
			basey00 = pyBas0[iter2];
			basex01 = basex00 + 1;
			basey01 = basey00;
			basex10 = basex00;
			basey10 = basey00 + 1;
			basex11 = basex00 + 1;
			basey11 = basey00 + 1;

			pImg1 = img.ptr<float>(basey00);
			pImg2 = img.ptr<float>(basey10);

			intensity00 = pImg1[basex00];
			intensity01 = pImg1[basex01];
			intensity10 = pImg2[basex10];
			intensity11 = pImg2[basex11];

			poutM[iter2] = pPerc00[iter2] * intensity00 + pPerc01[iter2] * intensity01 + pPerc10[iter2] * intensity10 + pPerc11[iter2] * intensity11;
		}
	}

	return outM;
}

cv::Mat warpping::floor(cv::Mat M)
{
	//**********************************************************
	//对Mat数据类型进行向下取整，返回CV_32S
	//输入：	M：为CV_32F Mat类型要取整的矩阵
	//**********************************************************

	cv::Mat outM(M.size(), CV_32S);
	//cv::Mat outM;
	int cols = M.cols;
	int rows = M.rows;
	for (int iter1 = 0;iter1 < rows;iter1++)
	{
		float* pM = M.ptr<float>(iter1);		//决定了输入矩阵必须为32F
		int* poutM = outM.ptr<int>(iter1);
		for (int iter2 = 0;iter2 < cols;iter2++)
		{
			poutM[iter2] = cvFloor(pM[iter2]);
		}
	}
	return outM;
}

cv::Mat warpping::floor(cv::Mat M, int max, int min)
{
	//**********************************************************
	//对Mat数据类型进行向下取整，同时进行上下限检测，返回CV_32S
	//输入：	M：为CV_32F Mat类型要取整的矩阵
	//**********************************************************

	cv::Mat outM(M.size(), CV_32S);
	//cv::Mat outM;
	int cols = M.cols;
	int rows = M.rows;
	for (int iter1 = 0;iter1 < rows;iter1++)
	{
		float* pM = M.ptr<float>(iter1);	//决定了输入矩阵必须为32F
		int* poutM = outM.ptr<int>(iter1);
		for (int iter2 = 0;iter2 < cols;iter2++)
		{
			poutM[iter2] = cvFloor(pM[iter2]);
			if (poutM[iter2]<min || poutM[iter2]>max)
			{
				poutM[iter2] = 0;	//当超过上下限，都置为0值
			}
		}
	}
	return outM;
}

cv::Mat warpping::ceil(cv::Mat M)
{
	//**********************************************************
	//对Mat数据类型进行向上取整，返回CV_32S
	//输入：	M：为CV_32F Mat类型要取整的矩阵
	//**********************************************************

	cv::Mat outM(M.size(), CV_32S);
	//cv::Mat outM;
	int cols = M.cols;
	int rows = M.rows;
	for (int iter1 = 0;iter1 < rows;iter1++)
	{
		float* pM = M.ptr<float>(iter1);
		int* poutM = outM.ptr<int>(iter1);
		for (int iter2 = 0;iter2 < cols;iter2++)
		{
			poutM[iter2] = cvCeil(pM[iter2]);
		}
	}
	return outM;
}

cv::Mat warpping::ceil(cv::Mat M, int max, int min)
{
	//**********************************************************
	//对Mat数据类型进行向上取整，同时进行上下限检测，返回CV_32S
	//输入：	M：为CV_32F Mat类型要取整的矩阵
	//**********************************************************

	cv::Mat outM(M.size(), CV_32S);
	//cv::Mat outM;
	int cols = M.cols;
	int rows = M.rows;
	for (int iter1 = 0;iter1 < rows;iter1++)
	{
		float* pM = M.ptr<float>(iter1);
		int* poutM = outM.ptr<int>(iter1);
		for (int iter2 = 0;iter2 < cols;iter2++)
		{
			poutM[iter2] = cvCeil(pM[iter2]);
			if (poutM[iter2]<min || poutM[iter2]>max)
			{
				poutM[iter2] = 0;	//当超过上下限，都置为0值
			}
		}
	}
	return outM;
}