
/*#include <stdio.h>
#include <iostream>
 
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/stitching/stitcher.hpp"

using namespace cv;

int main( int argc, char** argv )
{
	Stitcher stitcher = Stitcher::createDefault();
	Mat images[2];
	images[0] = imread( argv[2] );
	images[1] = imread( argv[1] );
	Mat pano;
	Stitcher::Status status = stitcher.composePanorama(images, pano);
	imwrite("Pano.jpg", pano);
}
*/

#include "stdio.h"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching/stitcher.hpp"
#ifdef _DEBUG 
#pragma comment(lib, "opencv_core246d.lib")  
#pragma comment(lib, "opencv_imgproc246d.lib")   //MAT processing 
#pragma comment(lib, "opencv_highgui246d.lib") 
#pragma comment(lib, "opencv_stitching246d.lib");
#else 
#pragma comment(lib, "opencv_core246.lib") 
#pragma comment(lib, "opencv_imgproc246.lib") 
#pragma comment(lib, "opencv_highgui246.lib") 
#pragma comment(lib, "opencv_stitching246.lib");
#endif 

using namespace cv; 
using namespace std;

int main() 
{
	 vector< Mat > vImg;

	 Mat rImg;

     vImg.push_back( imread("./img1.jpg") );
     vImg.push_back( imread("./img2.jpg") );
     vImg.push_back( imread("./img3.jpg") );
	 //vImg.push_back( imread("./stitching_img/S4.jpg") );
	 //vImg.push_back( imread("./stitching_img/S5.jpg") );
	 //vImg.push_back( imread("./stitching_img/S6.jpg") );
	 Stitcher stitcher = Stitcher::createDefault();
	 unsigned long AAtime=0, BBtime=0; //check processing time
	 AAtime = getTickCount(); //check processing time
	 Stitcher::Status status = stitcher.stitch(vImg, rImg);
	 BBtime = getTickCount(); //check processing time
	 printf("%.2lf sec \n",  (BBtime - AAtime)/getTickFrequency() ); //check processing time

	 if (Stitcher::OK == status)
	  //imshow("Stitching Result",rImg);
	  imwrite("Final1.jpg",rImg);
	  else
	  printf("Stitching fail.");
	 //waitKey(0);

	return 0;
}  
