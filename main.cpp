/********************************************************************************************/
/************************************** Harris角点检测 **************************************/
/********************************************************************************************/
/* 测试环境：VS2015 opencv310_vc14_x64_debug*/
#include<opencv2\opencv.hpp>
#include<iostream>
#include<sstream>
#include<string>
#include "warpping.h"
#include "matOperations.h"

//#include <cv.h>
//#include <highgui.h>
//#include <stdio.h>
//#pragma comment(lib,"opencv_ml2411.lib")

int main()
{
	/********************************************************************************************/
	/************************************** Harris角点检测 **************************************/
	/********************************************************************************************/
	cv::Mat iniImg;
	//std::string imgBathPath = "C:\\Users\\Happy\\Documents\\Visual Studio 2015\\Projects\\HappyKLT_Tracker20160114\\SeqImages\\01";
	std::string imgBathPath = "G:\\Visual Studio 2015\\HappyKLT_Tracker20160114\\SeqImages\\01";
	//std::string imgBathPath = "C:\\Users\\Happy\\Documents\\Visual Studio 2015\\Projects\\HappyKLT_Tracker20160114\\SeqImages2\\01";
	std::string imgSuffix = ".jpeg";
	//std::string imgSuffix = ".jpg";
	iniImg = cv::imread(imgBathPath + "33" + imgSuffix, 1);
	cv::Mat copy1IniImage;
	iniImg.copyTo(copy1IniImage);
	std::cout << "The size of the img is: " << iniImg.size() << std::endl << std::endl;

	cv::namedWindow("iniImg", 1);
	cv::imshow("iniImg", iniImg);

	//cv::cvtColor(iniImg, iniImg, CV_32F);//注意图像的编码，常用的是CV_8U和CV_32F（如果需要转的话要使用这句语句），记住，常查手册，看清函数支持的编码格式
	if (iniImg.channels() == 3)
	{
		cv::cvtColor(iniImg, iniImg, CV_BGR2GRAY);
	}

	//形成Sobel核
	/*double dx[9] = {-1,0,1,-2,0,2,-1,0,1};
	cv::Mat kernelx(3,3,CV_32F,dx);*/				//这种赋值也行，但要注意数据类型，如果是8U,那就有问题了，无正负之分，故应为32F
	//cv::Mat kernelx = (cv::Mat_<double>(3, 3) << -1 / 6, 0, 1 / 6, -1 / 6, 0, 1 / 6, -1 / 6, 0, 1 / 6);
	cv::Mat kernelx = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); //这里double和float任意，都可以与32F编码图像进行运算
	cv::Mat kernely;
	cv::transpose(kernelx, kernely);

	//std::cout << "The Kernel x is: " << std::endl << ' ' << kernelx << std::endl << std::endl;
	//std::cout << "The Kernel y is: " << std::endl << ' ' << kernely << std::endl << std::endl;

	//计算一阶图像
	cv::Mat Ix, Iy;
	cv::filter2D(iniImg, Ix, CV_32F, kernelx, cv::Point(-1, -1));	//注意这里千万不要为8U的，不然就没有正负了
	cv::filter2D(iniImg, Iy, CV_32F, kernely, cv::Point(-1, -1));
	std::cout << "The subsection of the iniImg are: " << std::endl << ' ' << iniImg(cv::Range(0, 20), cv::Range(0, 20)) << std::endl << std::endl;
	std::cout << "The subsection of the Ix are: " << std::endl << ' ' << Ix(cv::Range(0, 20), cv::Range(0, 20)) << std::endl << std::endl;
	//cv::Sobel(iniImg, Ix, CV_32F, 1, 0, 3);			//Sobel是专门的用来计算图像的一阶、二阶、三阶的微分图像（注意编码），其最后一个参数对应cornerHariss
	//cv::Sobel(iniImg, Iy, CV_32F, 0, 1, 3);			//中的ksize（取3，5，7），这里其实最好使用Sobel，此时不需要kernelx和kernely，只需调Sobel的参数即可
	cv::namedWindow("Ix", 1);
	cv::imshow("Ix", Ix);
	cv::namedWindow("Iy", 1);
	cv::imshow("Iy", Iy);
	cv::waitKey(5000);
	

	//计算二阶图像
	cv::Mat Ix2, Iy2, Ixy;
	cv::multiply(Ix, Ix, Ix2);
	cv::multiply(Iy, Iy, Iy2);
	cv::multiply(Ix, Iy, Ixy);
	cv::namedWindow("Ix2", 1);
	cv::imshow("Ix2", Ix2);
	cv::namedWindow("Iy2", 1);
	cv::imshow("Iy2", Iy2);
	cv::waitKey(5000);

	//高斯加权（高斯模糊，高斯滤波）
	int ksizeBlur = 11;								//高斯滤波核的大小似乎对检测结果不太敏感
	double sigmax = 1;
	double sigmay = 1;
	cv::Mat A, B, C;
	cv::GaussianBlur(Ix2, A, cv::Size(ksizeBlur, ksizeBlur), sigmax, sigmay);
	cv::GaussianBlur(Iy2, B, cv::Size(ksizeBlur, ksizeBlur), sigmax, sigmay);
	cv::GaussianBlur(Ixy, C, cv::Size(ksizeBlur, ksizeBlur), sigmax, sigmay);
	cv::namedWindow("A", 1);
	cv::imshow("A", A);
	cv::namedWindow("B", 1);
	cv::imshow("B", B);
	cv::waitKey(5000);

	//计算响应函数矩阵R
	cv::Mat R, AB, C2, detM, traceM;
	cv::multiply(A, B, AB);
	cv::multiply(C, C, C2);
	cv::subtract(AB, C2, detM);
	double alpha = 0.04;
	cv::add(A, C, traceM);
	cv::subtract(detM, alpha*traceM.mul(traceM), R);
	cv::normalize(R, R, 1, 0, cv::NORM_MINMAX, CV_32F, cv::Mat()); //归一化（注意编码），不然R元素值太大，不好设定响应值阈值threR进行过滤（非必要，但是最好要）
	/*这里选用NORM_MINMAX，归一化原理是（R(x,y)-min(R)）/（max(R)-min(R)）*/
	//std::cout << "The subsection of the R are: " << std::endl << ' ' << R(cv::Range(180, 200), cv::Range(250, 270)) << std::endl << std::endl;

	//提取角点坐标：先膨胀，再二值化，然后求位与比较运算，再进行非极大值抑制，最后循环提取坐标
	cv::Mat dilated;		//膨胀：邻域内所有元素都去邻域内的极大值，也即只有极大值值保持不变，其他都会变化
	cv::Mat kernelDilate;
	//int ksizeDilate = 21;
	int ksizeDilate = 51;	//该邻域的大小很重要，很大程度上决定了所检测到的角点的数量，越大，检测到的角点数越少（对应cornerHarris函数的blocksize参数）
	kernelDilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksizeDilate, ksizeDilate));	//获取膨胀邻域，邻域都是这么获取的
	cv::dilate(R, dilated, kernelDilate, cv::Point(-1, -1));		//we can input cv::Mat() if we use the default value of the third parameter
	cv::Mat localMax;
	cv::compare(R, dilated, localMax, CV_CMP_EQ);	//通过与膨胀前图像的与比较得到个邻域内的极大值点（膨胀后只有各邻域的极大值点值不变，若两个值相等，则取1，否则取0）
	//std::cout << "The subsection of the localMax are: " << std::endl << ' ' << localMax(cv::Range(180, 200), cv::Range(250, 270)) << std::endl << std::endl;

	//非极大值抑制
	double threR = 0.01; //therR为响应值的阈值，只有大于这个阈值的极大值点才是真正的角点，这样可以保证角点品质，该值越大，角点越少，如果前面没有进行R矩阵归一化，则很不敏感，因此有需要进行归一化
	cv::Mat cornerTh, cornerMap;
	cv::threshold(R, cornerTh, threR, 255, cv::THRESH_BINARY);		//二值化，大于阈值取255，小于取0
	cornerTh.convertTo(cornerMap, CV_8U);							//注意编码转变，下面位与运算只支持8U的图像
	cv::bitwise_and(cornerMap, localMax, cornerMap);	//位与运算，值相同的取1，不同为0（只有同时满足邻域极大值和大于R阈值的点才是满足要求的角点，双重要求，可以任意去掉其中的一重，只是减小了角点过滤力度）
	//std::cout << "The subsection of the cornerMap are: " << std::endl << ' ' << cornerMap(cv::Range(180, 200), cv::Range(250, 270)) << std::endl << std::endl;

	//循环提取角点坐标，存入corners角点容器，注意编码
	std::vector<cv::Point2f> corners;
	std::vector<cv::Point2f> trackedCorners;
	for (int it_row = 0;it_row < cornerMap.rows;it_row++)		//.rows not row
	{
		uchar* plocalMax = cornerMap.ptr<uchar>(it_row);
		for (int it_col = 0;it_col < cornerMap.cols;it_col++)	//.cols not col
		{
			if (plocalMax[it_col])
			{
				corners.push_back(cv::Point2f((float)it_col, (float)it_row));
				cv::circle(copy1IniImage, cv::Point2f((float)it_col, (float)it_row), 2, cv::Scalar(0, 255, 0), -1, 8, 0);	//标记角点

				if ((it_col == 190 && it_row == 149) || (it_col == 179 && it_row == 188))
				//if ((it_col == 409 && it_row == 181) || (it_col == 331 && it_row == 256))
				{
					//这里人工挑出在人身上的两个跟踪点
					trackedCorners.push_back(cv::Point2f((float)it_col, (float)it_row));
					cv::circle(copy1IniImage, cv::Point2f((float)it_col, (float)it_row), 6, cv::Scalar(0, 0, 255), 2, 8, 0);
				}
			}
		}
	}

	std::cout << "The total numer of the detected Harris corner is: " << corners.size() << std::endl;
	std::cout << "The detected Harris corner are: " << std::endl << ' ' << corners << std::endl << std::endl;
	cv::namedWindow("Harris Result", 1);
	cv::imshow("Harris Result", copy1IniImage);
	cv::waitKey(2000);

	/********************************************************************************************/
	/*************************************** KLT Tracker ****************************************/
	/********************************************************************************************/
	cv::Mat preImg, nextImg, copy1NextImg;			//前一阵图像、下一帧图像以及下帧图像的拷贝
	cv::Mat nextIx, nextIy;							//下针图像的x方向梯度图像，y方向梯度方向
	std::vector<cv::Point2f> preCorner, nextCorner;	//前一帧图像中的跟踪点位置，下一帧图像中的跟踪点位置（二维）

	iniImg.copyTo(preImg);							//初始帧赋值
	preCorner = trackedCorners;						//初始跟踪点位置赋值

	int imgRow = iniImg.rows;						//图像的行数
	int imgCol = iniImg.cols;						//图像的行数

	float tempx_s, tempx_a, tempy_s, tempy_a;		//所用窗口的行列上下限（用来检查，防止窗口到图像外部）
	int sobelSize = 3;								//求梯度图像中所用的Soble核尺寸
	int nPoint = trackedCorners.size();				//跟踪点的个数
	std::cout << "The number of tracked corners is: " << nPoint << std::endl << std::endl;

	std::stringstream Num;							//字符串流
	std::string imgNum, fullPath;					//图片编号，路径全名
	int num = 34;									//除去初始的图片，剩余的图片序列从34开始编号

	//int templateSize = 3;							//模板和窗口图像的尺寸（模板只上帧图像跟踪点所在的矩形区域，可按行列方位提取）
	int templateSize = 10;
	int windowSize = 2 * templateSize + 1;			//实际模板和窗口图像的大小为2*templateSize + 1（窗口是指下帧图像跟踪点所在区域，不可按行列号提取，因为经过了旋转，区域有歪斜）

	//if (templateSize % 2 != 1) templateSize++;	//奇数模板尺寸

	cv::Mat matX, matY;								//窗口（或叫mask），分别存储窗口的行和列号
	matX = matOperations::seqMatCol(2 * templateSize + 1, 2 * templateSize + 1);	//CV_32S，形成列号矩阵（存储窗口的列号）
	matY = matOperations::seqMatRow(2 * templateSize + 1, 2 * templateSize + 1);	//CV_32S，形成行号矩阵（存储窗口的行号）

	matX = matX - templateSize;						//使得窗口中心的坐标为（0，0）
	matY = matY - templateSize;						//其他的坐标按x（水平向右）,y（竖直向下）取相对坐标

	cv::Mat templateImg, nextWarpped, nextROIx, nextROIy;		//上帧图像中提取的模板，下帧图像经仿射变换提取出的窗口，以及x向梯度，y向梯度
	cv::Mat err(windowSize, windowSize, CV_32F);	//前后阵跟踪点区域灰度值之差（即模板灰度值-窗口灰度值）
	//cv::Mat JacobiW(2, 6, CV_32F);				//W(x;p)梯度
	//cv::Mat Hessi(6, 6, CV_32F);					//Hessian矩阵，该矩阵的大小由P的大小决定，本算法P的大小为6，故Hessian为6*6
	cv::Mat GradientXY;								//nextImg在x,y处的梯度，1*2维
	cv::Mat JacobiW;								//W(x;p)梯度，变换矩阵在x,y处的一阶倒数值，对应第四步计算结果
	cv::Mat steepestDesc;							//最速下降值（对应第五步计算结果）
	cv::Mat TsteepestDesc;							//steepestDesc的转置
	cv::Mat Hessi;									//Hessian矩阵（对应第六步结算结果），该矩阵的大小由P的大小决定，本算法P的大小为6，故Hessian为6*6
	cv::Mat invHessi;								//Hessian矩阵的逆阵
	cv::Mat totalDesc;								//下降值（对应第七步计算结果）
	cv::Mat deltaP;									//对应第八步结算结果

	std::vector<cv::Point2f> cornerTrack1;			//存储被跟踪点（1号）的轨迹，这里只跟踪两个点，因此，只有两个vector容器
	std::vector<cv::Point2f> cornerTrack2;			//存储被跟踪点（2号）的轨迹，这里只跟踪两个点，因此，只有两个vector容器
	cv::Point2f newCorner;							//用来存储跟踪点的新坐标

	double eps = 0.01;								//梯度下降法终止精度（终止条件）
	double iter;									//迭代次数

	////仿射（warp）变换参数（初始值取跟踪点的初始坐标）
	float P[2][6] = { {0,0,0,0,preCorner[0].x ,preCorner[0].y },{ 0,0,0,0,preCorner[1].x ,preCorner[1].y } }; 

	//如果采用高斯滤波求梯度图像，则需要使用以下高斯核
	//cv::Mat DGaussX = matOperations::gaussianKernelDeriX(3.0);
	//cv::Mat DGaussY = matOperations::gaussianKernelDeriY(3.0);
	//std::cout << "The DGaussX is: " << std::endl << DGaussX << std::endl << std::endl;
	//std::cout << "The DGaussY is: " << std::endl << DGaussY << std::endl << std::endl;
	//cv::Mat flipDGaussX, flipDGaussY;
	//cv::flip(DGaussX, flipDGaussX, -1);
	//cv::flip(DGaussY, flipDGaussY, -1);

	//依据序列图像的帧数进行循环
	for (num;num<88;num++)
	{
		//读取下帧图像
		Num << num;		
		Num >> imgNum;
		Num.clear();

		fullPath = imgBathPath + imgNum + imgSuffix;
		nextImg = cv::imread(fullPath, 1);
		nextImg.copyTo(copy1NextImg);

		//图像读取失败，跳出循环
		if (nextImg.empty()) break;

		//读取图片若为3通道，转换成单通道
		if (nextImg.channels() == 3)	cv::cvtColor(nextImg, nextImg, CV_BGR2GRAY);

		//每读进一帧图像，需计算所有跟踪点的位置，因此for循环的次数为nPoint（跟踪点个数）
		for (int iter1 = 0;iter1 < nPoint;iter1++)
		{
			deltaP = cv::Mat::ones(6, 1, CV_32F);				//deltaP赋初值
			iter = 0;

			tempx_s = preCorner[iter1].x - templateSize;
			tempx_a = preCorner[iter1].x + templateSize;
			tempy_s = preCorner[iter1].y - templateSize;
			tempy_a = preCorner[iter1].y + templateSize;

			if ((tempx_a >= 0) && (tempx_a < imgCol) && (tempy_s >= 0) && (tempy_a < imgRow))
			{
				//迭代循环，求跟踪点在下帧图像中的坐标
				while (cv::norm(deltaP) > eps)
				{
					iter++;
					if (iter > 80) break;							//迭代次数超过80还没达到精度，则也退出

					cv::Mat transformMatrix = (cv::Mat_<float>(2, 3) << 1 + P[iter1][0], P[iter1][2], P[iter1][4], P[iter1][1], 1 + P[iter1][3], P[iter1][5]);

					//随着迭代，tempy_s等会出现小数（像素坐标为小数），Range会自动对其向下取整
					preImg(cv::Range(tempy_s, tempy_a + 1), cv::Range(tempx_s, tempx_a + 1)).copyTo(templateImg);

					//1.对下帧图像进行仿射变换，得到窗口
					nextWarpped = warpping::warpTransform(nextImg, transformMatrix, matX, matY);	//输入图像为CV_32F，不需转换（注意matX和matY不要弄反）
					//opencv2里面也有放射变换函数AffineWarp，用这个函数需要逆向线性插值

					//2.计算模板和窗口的灰度差
					templateImg.convertTo(templateImg, CV_32F);		//转成CV_32F，下面要与CV_32F的nextWarpped作减法（不转会出错）
					//nextWarpped.convertTo(nextWarpped, CV_32F);		
					cv::subtract(templateImg, nextWarpped, err);	//之所以不再定义的时候就定义成CV_32F，是因为有些时候，刚好是整数，opencv中会自动转换类型成CV_8U之类的，所以没办法只得每次都转

					//3.对下针图像先进行一阶梯度图像计算，然后进行放射变换得到窗口的x，y向梯度图
					//cv::Mat filterKernelx = (cv::Mat_<float>(1,3) << -1,0, 1);
					//cv::Mat filterKernely;
					//cv::transpose(filterKernelx, filterKernely);
					//cv::filter2D(nextImg, nextIx, CV_32F, filterKernelx);
					//cv::filter2D(nextImg, nextIy, CV_32F, filterKernely);

					//Sobel算子求梯度图像
					cv::Sobel(nextImg, nextIx, CV_32F, 1, 0, sobelSize);
					cv::Sobel(nextImg, nextIy, CV_32F, 0, 1, sobelSize);

					//高斯滤波求梯度图像
					//cv::filter2D(nextImg, nextIx, CV_32F, flipDGaussX);
					//cv::filter2D(nextImg, nextIy, CV_32F, flipDGaussY);

					//std::cout << "The nextIx is: " << std::endl << ' ' << nextIx(cv::Range(100, 120), cv::Range(100, 120)) << std::endl << std::endl;
					//std::cout << "The nextIy is: " << std::endl << ' ' << nextIy(cv::Range(100, 120), cv::Range(100, 120)) << std::endl << std::endl;
					//cv::namedWindow("DX", 1);
					//cv::namedWindow("DY", 1);
					//cv::imshow("DX", nextIx);
					//cv::imshow("DY", nextIy);
					//cv::waitKey(10000);
					//cv::GaussianBlur(nextImg, nextIx, cv::Size(ksizeBlur + 8, ksizeBlur + 8), 1, 0);
					//cv::GaussianBlur(nextImg, nextIy, cv::Size(ksizeBlur + 8, ksizeBlur + 8), 0, 1);

					nextROIx = warpping::warpTransform(nextIx, transformMatrix, matX, matY);	//输出即为CV_32F，不需转换
					nextROIy = warpping::warpTransform(nextIy, transformMatrix, matX, matY);
					//nextROIx.convertTo(nextROIx, CV_32F);		
					//nextROIy.convertTo(nextROIy, CV_32F);

					//4.5.6.7.就是变换矩阵的雅克比矩阵，计算最速下降图像，计算海塞矩阵，计算最优下降幅度，由于按每个像素计算，故这几步杂合在循环中
					Hessi = cv::Mat::zeros(6, 6, CV_32F);		//初始值为0阵
					totalDesc = cv::Mat::zeros(6, 1, CV_32F);	//初始值为0阵，其维数为（6*1）*1

					matX.convertTo(matX, CV_32F);				//转换为CV_32F，下面计算都用这个类型
					matY.convertTo(matY, CV_32F);

					//按像素挨个计算：先按行，在行列，循环次数为模板或窗口的总像素个数
					for (int iter2 = 0;iter2 < 2 * templateSize + 1;iter2++)
					{
						float* pMatX = matX.ptr<float>(iter2);				//获取列号矩阵列指针
						float* pMatY = matY.ptr<float>(iter2);				//获取行号矩阵行指针
						float* pNextROIx = nextROIx.ptr<float>(iter2);		//获取窗口x梯度图像行指针
						float* pNextROIy = nextROIy.ptr<float>(iter2);		//获取窗口y梯度图像行指针

						float* pErr = err.ptr<float>(iter2);				//获取误差图像行指针

						for (int iter3 = 0;iter3 < 2 * templateSize + 1;iter3++)
						{
							GradientXY = (cv::Mat_<float>(1, 2) << pNextROIx[iter3], pNextROIy[iter3]);		//（x,y）像素处的梯度

							//4.计算（x,y）像素处的变换矩阵雅克比矩阵
							JacobiW = (cv::Mat_<float>(2, 6) << pMatX[iter3], 0, pMatY[iter3], 0, 1, 0,		
								0, pMatX[iter3], 0, pMatY[iter3], 0, 1);

							//5.计算（x,y）像素处最速下降图像
							steepestDesc = GradientXY*JacobiW;					
							cv::transpose(steepestDesc, TsteepestDesc);			//转置

							//6.求海塞矩阵，这里乘法不能用.mul或者multiply，这里是矩阵乘法，不是点乘
							cv::add(Hessi, TsteepestDesc*steepestDesc, Hessi);	

							//7.计算总下降值
							cv::add(totalDesc, TsteepestDesc*pErr[iter3], totalDesc);
						}
					}

					//8.求deltaP
					cv::invert(Hessi, invHessi);
					deltaP = invHessi*totalDesc;

					//.9更新P
					for (int iter4 = 0;iter4 < 6;iter4++)
					{
						float* pTdeltaP = deltaP.ptr<float>(iter4);		//deltaP是一个6*1的列向量
						P[iter1][iter4] = P[iter1][iter4] + pTdeltaP[0];
					}
				}
			}
			
			std::cout << "The iter is: " << iter << std::endl << std::endl;
			nextCorner.push_back(cv::Point2f(P[iter1][4], P[iter1][5]));

			//将跟踪点的新坐标存入轨迹数组（用作回放图像）
			if (iter1 == 0) cornerTrack1.push_back(nextCorner[iter1]);
			else cornerTrack2.push_back(nextCorner[iter1]);
		}

		//判断跟踪点的新坐标是否在图像内，否则判断为跟丢
		if (nextCorner[0].x < 0 || nextCorner[0].x > imgCol || nextCorner[0].y < 0 || nextCorner[0].y > imgRow)
		{
			std::cout << "The first tracked corner is lost!!!" << std::endl;
			std::cout << "The first corner is: " << nextCorner[0].x << ',' << nextCorner[0].y << std::endl << std::endl;
			break;
		}
		if (nextCorner[1].x < 0 || nextCorner[1].x > imgCol || nextCorner[1].y < 0 || nextCorner[1].y > imgRow)
		{
			std::cout << "The second tracked corner is lost!!!" << std::endl;
			std::cout << "The second corner is: " << nextCorner[1].x << ',' << nextCorner[1].y << std::endl << std::endl;
			break;
		}

		cv::circle(copy1NextImg, nextCorner[0], 5, cv::Scalar(0, 255, 0), 2);
		cv::circle(copy1NextImg, nextCorner[1], 5, cv::Scalar(0, 255, 0), 2);

		//更新跟踪点的坐标
		preCorner = nextCorner;
		//新除nextCorner，因为这个数组值存储当前的新坐标
		nextCorner.clear();
		//下帧图像变为上帧图像，进行下次循环
		nextImg.copyTo(preImg);

		cv::namedWindow("Track Result", 1);
		cv::imshow("Track Result", copy1NextImg);
		cv::waitKey(10);
	}
	system("pause");
}