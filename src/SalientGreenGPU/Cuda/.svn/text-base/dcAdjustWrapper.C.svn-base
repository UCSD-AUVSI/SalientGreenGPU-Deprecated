


void dcAdjustWrapper( cv::gpu::GpuMat const & mat )
{
	int height = mat.

}
//! Adds to the DC offset for a matrix
/*! @param[in] mat The CUDA device memory for the matrix
	  @param[in] value The value to add to the DC component */
__global__ dcAdjust( cv::gpu::DevMem2Df mat, float value )
{
	int row = 0, col = 0;
	mat.ptr( row )[col] += value;
}
