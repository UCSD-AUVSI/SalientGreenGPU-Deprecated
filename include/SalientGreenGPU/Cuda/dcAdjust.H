
#include <opencv2/gpu/devmem2d.hpp>

struct myfloat2
{
	float x;
	float y;
};

__declspec(dllimport) void addFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

__declspec(dllimport) void addAndZeroFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );

__declspec(dllimport) void subFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

__declspec(dllimport) void mulFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

__declspec(dllimport) void mulValueFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, float value,
		cv::gpu::DevMem2D_<myfloat2> c );

//! Adjusts the dc component of some matrix
/*! @param[in] mat Device memory for the matrix (float)
 		@param[in] height height of the matrix
		@param[in] width width of the matrix
		@param[in] value the value to add */
__declspec(dllimport) void dcAdjustWrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );


__declspec(dllimport) void absFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat );

__declspec(dllimport) void maxFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );


__declspec(dllimport) void minMaxHelperWrapper( cv::gpu::DevMem2D_<float> const mat, float thresh, float & sum, int & count );
