#pragma once
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>

#include<opencv2\opencv.hpp>
#include <cv.h>
#include <highgui.h>


using namespace std;

/*----------------------------
* 功能 : 将 cv::Mat 数据写入到 .txt 文件
*----------------------------
* 函数 : WriteData
* 访问 : public
* 返回 : -1：打开文件失败；0：写入数据成功；1：矩阵为空
*
* 参数 : fileName	[in]	文件名
* 参数 : matData	[in]	矩阵数据
*/
int WriteData(string fileName, cv::Mat& matData)
{
	int retVal = 0;

	// 检查矩阵是否为空
	if (matData.empty())
	{
		cout << "矩阵为空" << endl;
		retVal = 1;
		return (retVal);
	}

	// 打开文件
	ofstream outFile(fileName.c_str(), ios_base::out);	//按新建或覆盖方式写入
	if (!outFile.is_open())
	{
		cout << "打开文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 写入数据
	for (int r = 0; r < matData.rows; r++)
	{
		for (int c = 0; c < matData.cols; c++)
		{
			int data = matData.at<uchar>(r, c);	//读取数据，at<type> - type 是矩阵元素的具体数据格式
			outFile << data << "\t";	//每列数据用 tab 隔开
		}
		outFile << endl;	//换行
	}

	return (retVal);
}


/*----------------------------
* 功能 : 从 .txt 文件中读入数据，保存到 cv::Mat 矩阵
*		- 默认按 float 格式读入数据，
*		- 如果没有指定矩阵的行、列和通道数，则输出的矩阵是单通道、N 行 1 列的
*----------------------------
* 函数 : LoadData
* 访问 : public
* 返回 : -1：打开文件失败；0：按设定的矩阵参数读取数据成功；1：按默认的矩阵参数读取数据
*
* 参数 : fileName	[in]	文件名
* 参数 : matData	[out]	矩阵数据
* 参数 : matRows	[in]	矩阵行数，默认为 0
* 参数 : matCols	[in]	矩阵列数，默认为 0
* 参数 : matChns	[in]	矩阵通道数，默认为 0
*/
int LoadData(string fileName, cv::Mat& matData, int matRows = 0, int matCols = 0, int matChns = 0)
{
	int retVal = 0;

	// 打开文件
	ifstream inFile(fileName.c_str(), ios_base::in);
	if (!inFile.is_open())
	{
		cout << "读取文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 载入数据
	istream_iterator<float> begin(inFile);	//按 float 格式取文件数据流的起始指针
	istream_iterator<float> end;			//取文件流的终止位置
	vector<float> inData(begin, end);		//将文件数据保存至 std::vector 中
	cv::Mat tmpMat = cv::Mat(inData);		//将数据由 std::vector 转换为 cv::Mat

											// 输出到命令行窗口
											//copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t")); 

											// 检查设定的矩阵尺寸和通道数
	size_t dataLength = inData.size();
	//1.通道数
	if (matChns == 0)
	{
		matChns = 1;
	}
	//2.行列数
	if (matRows != 0 && matCols == 0)
	{
		matCols = dataLength / matChns / matRows;
	}
	else if (matCols != 0 && matRows == 0)
	{
		matRows = dataLength / matChns / matCols;
	}
	else if (matCols == 0 && matRows == 0)
	{
		matRows = dataLength / matChns;
		matCols = 1;
	}
	//3.数据总长度
	if (dataLength != (matRows * matCols * matChns))
	{
		cout << "读入的数据长度 不满足 设定的矩阵尺寸与通道数要求，将按默认方式输出矩阵！" << endl;
		retVal = 1;
		matChns = 1;
		matRows = dataLength;
	}

	// 将文件数据保存至输出矩阵
	matData = tmpMat.reshape(matChns, matRows).clone();

	return (retVal);
}
