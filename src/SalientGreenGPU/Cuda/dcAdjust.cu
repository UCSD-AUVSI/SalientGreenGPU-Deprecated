
#include <SalientGreenGPU/Cuda/dcAdjust.H>
  
__global__ void minMaxHelper( cv::gpu::DevMem2D_<float> const mat, float thresh, float * sum, int * count )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x > 0 && x < mat.cols - 1 && y > 0 && y < mat.rows - 1 )
	{
		const float value = mat.ptr( y )[x];
		const float neighborLeft = mat.ptr( y )[x - 1];
		const float neighborUp = mat.ptr( y - 1 )[x];
		const float neighborDown = mat.ptr( y + 1 )[x];
		const float neighborRight = mat.ptr( y )[x + 1];

		if( value >= thresh &&
				value > neighborLeft &&
				value > neighborUp &&
				value > neighborDown &&
				value >= neighborRight )
		{
			*sum += value;
			*count = *count + 1;
		}
	}
}
__global__ void absFC2( cv::gpu::DevMem2D_<myfloat2> mat )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < mat.cols && y < mat.rows )
	{
		mat.ptr( y )[x].x = fabsf( mat.ptr( y )[x].x );
		mat.ptr( y )[x].y = fabsf( mat.ptr( y )[x].y );
	}
}

__global__ void maxFC2( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < mat.cols && y < mat.rows )
	{
		mat.ptr( y )[x].x = fmaxf( mat.ptr( y )[x].x, value );
		mat.ptr( y )[x].y = fmaxf( mat.ptr( y )[x].y, value );
	}
}

//! Adds to the DC offset for a matrix
/*! @param[in] mat The CUDA device memory for the matrix
	  @param[in] value The value to add to the DC component */
__global__ void dcAdjust( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	mat.ptr( 0 )[0].x += value;
}

//! computes c = a + b
__global__ void addFC2( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x + b.ptr( y )[x].x;
		c.ptr( y )[x].y = a.ptr( y )[x].y + b.ptr( y )[x].y;
	}
}

__global__ void addAndZeroFC2( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < mat.cols && y < mat.rows )
	{
		mat.ptr( y )[x].x += value; fmaxf( mat.ptr( y )[x].x, value );
		mat.ptr( y )[x].y = 0.0f;
	}
}

//! computes c = a - b
__global__ void subFC2( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x - b.ptr( y )[x].x;
		c.ptr( y )[x].y = a.ptr( y )[x].y - b.ptr( y )[x].y;
	}

}
//! computes c = a * b
__global__ void mulFC2( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x * b.ptr( y )[x].x;
		c.ptr( y )[x].y = a.ptr( y )[x].y * b.ptr( y )[x].y;
	}

}

//! computes c = a * b
__global__ void mulValueFC2( cv::gpu::DevMem2D_<myfloat2> const a, float const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x * b;
		c.ptr( y )[x].y = a.ptr( y )[x].y * b;
	}

}

static inline int divUp( int total, int grain )
{
	return ( total + grain - 1 ) / grain;
}

void addFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	addFC2<<< grids, threads >>>( a, b, c );
}

void addAndZeroFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	addAndZeroFC2<<< grids, threads >>>( mat, value );
}
void subFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	subFC2<<< grids, threads >>>( a, b, c );
}

void mulFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	mulFC2<<< grids, threads >>>( a, b, c );
}

void mulValueFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, float const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	mulValueFC2<<< grids, threads >>>( a, b, c );
}

void dcAdjustWrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	dcAdjust<<< grids, threads >>>( mat, value );
}


void absFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	absFC2<<< grids, threads >>>( mat );
}

void maxFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	maxFC2<<< grids, threads >>>( mat, value );
}

void minMaxHelperWrapper( cv::gpu::DevMem2D_<float> const mat, float thresh, float & sum, int & count )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	float * dSum;
	int * dCount;
	cudaMalloc(&dSum, sizeof(float));
	cudaMalloc(&dCount, sizeof(int));

	minMaxHelper<<< grids, threads >>>( mat, thresh, dSum, dCount );
	cudaMemcpy( &sum, dSum, sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( &count, dCount, sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree( dCount );
	cudaFree( dSum );
}
