
#include <opencv2/gpu/devmem2d.hpp>

struct myfloat2
{
	float x;
	float y;
};

void addFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

void addAndZeroFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );

void subFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

void mulFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

void mulValueFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, float value,
		cv::gpu::DevMem2D_<myfloat2> c );

//! Adjusts the dc component of some matrix
/*! @param[in] mat Device memory for the matrix (float)
 		@param[in] height height of the matrix
		@param[in] width width of the matrix
		@param[in] value the value to add */
void dcAdjustWrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );


void absFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat );

void maxFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );


void minMaxHelperWrapper( cv::gpu::DevMem2D_<float> const mat, float thresh, float & sum, int & count );
