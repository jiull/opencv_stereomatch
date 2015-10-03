// BM_SGBM_GC.cpp: 主项目文件。

#include "stdafx.h"
#include <opencv.hpp>
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <iostream>
using namespace std;
using namespace cv;

int main()
{

		IplImage * left_rectified = cvLoadImage( "imL.jpg", 0);
		IplImage * right_rectified= cvLoadImage( "imR.jpg", 0);
		CvSize imageSize = cvGetSize( left_rectified );

		IplImage * left_rgb= cvLoadImage( "imL.jpg", 1);
	    IplImage * right_rgb= cvLoadImage( "imR.jpg", 1);
///////////////////////////////OpenCV_BM立体匹配///////////////////////////////////////
CvMat* raw_disp = cvCreateMat( imageSize.height,imageSize.width, CV_16S );
CvMat* BM_disparity = cvCreateMat( imageSize.height,imageSize.width, CV_8U );
CvStereoBMState *BMState = cvCreateStereoBMState();
int SADWindowSize=19; 
 BMState->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
 BMState->minDisparity = 0;
 BMState->numberOfDisparities = 32;
 BMState->textureThreshold = 10;
 BMState->uniquenessRatio = 10;
 BMState->speckleWindowSize = 17;
 BMState->speckleRange = 32;
 BMState->disp12MaxDiff = 1;
 cvFindStereoCorrespondenceBM( left_rectified,right_rectified, raw_disp,BMState);
 cvNormalize( raw_disp, BM_disparity, 0, 256, CV_MINMAX );
//////////////////////////////////OpenCV_GC立体匹配////////////////////////////////////////
CvMat* disparity_left = cvCreateMat( imageSize.height, imageSize.width, CV_16S );
CvMat* disparity_right = cvCreateMat( imageSize.height, imageSize.width, CV_16S );
CvStereoGCState* state = cvCreateStereoGCState( 64, 3 );
cvFindStereoCorrespondenceGC( left_rectified,
  right_rectified,
  disparity_left,
  disparity_right,
  state,
  0 );
cvReleaseStereoGCState( &state );
CvMat* GC_disparity = cvCreateMat( imageSize.height, imageSize.width, CV_8U );
cvConvertScale( disparity_left, GC_disparity, -8 );
//////////////////////////////OpenCV_SGBM立体匹配////////////////////////////////
cv::StereoSGBM sgbm;  
        sgbm.preFilterCap = 63;  
        int SGBM_SADWindowSize=11; 
		/*int numberOfDisparities=numberOfDisparities > 0 ? numberOfDisparities : ((imageSize.width/8) + 15) & -16;*/
		int numberOfDisparities=32;
        int cn = left_rectified->nChannels;  
        sgbm.SADWindowSize = SGBM_SADWindowSize > 0 ? SGBM_SADWindowSize : 3;  
        sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;  
        sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;  
        sgbm.minDisparity = 0;  
        sgbm.numberOfDisparities = numberOfDisparities;  
        sgbm.uniquenessRatio = 10;  
        sgbm.speckleWindowSize = 100;  
        sgbm.speckleRange = 32;  
        sgbm.disp12MaxDiff = 1; 

		Mat disp, SGBM_disparity;
        sgbm((Mat)left_rectified, (Mat)right_rectified,disp );  
		disp.convertTo(SGBM_disparity, CV_8U, 255/(numberOfDisparities*16.));
//创建窗口显示
cvNamedWindow( "left_rectified" );
cvNamedWindow( "right_rectified" );
cvShowImage("left_rectified", left_rgb );
cvShowImage("right_rectified", right_rgb );
cvNamedWindow( "GC_Disparity");
cvShowImage("GC_Disparity", GC_disparity );
cvSaveImage( "GC_Disparity.jpg", GC_disparity );
cvNamedWindow( "BM_Disparity");
cvShowImage("BM_Disparity", BM_disparity );
cvSaveImage( "BM_Disparity.jpg", BM_disparity );
namedWindow("SGBM_Disparity", 1);
imshow("SGBM_Disparity", SGBM_disparity);
imwrite("SGBM_Disparity.jpg", SGBM_disparity); 

cvWaitKey( 0 );
cvReleaseImage(&left_rectified);
cvReleaseImage(&right_rectified);
cvReleaseMat(&disparity_left);
cvReleaseMat(&GC_disparity);
cvDestroyWindow( "GC_Disparity" );
return 0;
}

