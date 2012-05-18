/*
 * Salient Green
 * Copyright (C) 2011 Shane Grant
 * wgrant@usc.edu
 *
 * Salient Green is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 3.0
 * of the License, or (at your option) any later version.
 *
 * Salient Green is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Salient Green; if not, see
 * <http://www.gnu.org/licenses/> */

#include <SalientGreenGPU/Filters/Utility.H>
#include <opencv2/imgproc/imgproc.hpp>

void sg::fftshift( cv::Mat & src )
{
  // If even we can just pivot around the center pixel
  // if odd, we pivot around center + (1,1)
  // quadrants arranged q0 q1
  //                    q2 q3
  // q0 is located at (0, 0) with width px, height py
  // q1 is located at (px, 0) with width hw, height py
  // q2 is located at (0, py) with width px, height hw
  // q3 is located at (px, py) with width hw, height hw
  int hwx = src.cols / 2;
  int hwy = src.rows / 2;
  int px = hwx;
  int py = hwy;

  if( src.cols %2 != 0 )
    ++px;

  if( src.rows % 2 != 0 )
    ++py;

  cv::Mat  q0( src, cv::Rect( 0, 0, px, py ) );
  cv::Mat q02( src, cv::Rect( hwx, hwy, px, py ) );

  cv::Mat  q1( src, cv::Rect( px, 0, hwx, py ) );
  cv::Mat q12( src, cv::Rect( 0, hwy, hwx, py ) );

  cv::Mat  q2( src, cv::Rect( 0, py, px, hwy ) );
  cv::Mat q22( src, cv::Rect( hwx, 0, px, hwy ) );

  cv::Mat  q3( src, cv::Rect( px, py, hwx, hwy ) );
  cv::Mat q32( src, cv::Rect( 0, 0, hwx, hwy ) );

  // q0 is the biggest segment and might overlap with
  // everything else so it gets done last, we hold it in tmp
  cv::Mat tmpq0, tmpq2;
  q0.copyTo( tmpq0 );
  q2.copyTo( tmpq2 );

  q3.copyTo( q32 );
  q1.copyTo( q12 );
  tmpq2.copyTo( q22 );
  tmpq0.copyTo( q02 );
}

void sg::ifftshift( cv::Mat & src )
{
  // Our pivot is always the center pixel
  // quadrants arranged q0 q1
  //                    q2 q3
  // q0 is located at (0, 0) with width hw, height hw
  // q1 is located at (hw, 0) with width px, height hw
  // q2 is located at (0, hw) with width hw, height py
  // q3 is located at (hw, hw) with width px, height py
  int hwx = src.cols / 2;
  int hwy = src.rows / 2;
  int px = hwx;
  int py = hwy;

  if( src.cols %2 != 0 )
    ++px;

  if( src.rows % 2 != 0 )
    ++py;

  cv::Mat  q0( src, cv::Rect( 0, 0, hwx, hwy ) );
  cv::Mat q02( src, cv::Rect( px, py, hwx, hwy ) );

  cv::Mat  q1( src, cv::Rect( hwx, 0, px, hwy ) );
  cv::Mat q12( src, cv::Rect( 0, py, px, hwy ) );

  cv::Mat  q2( src, cv::Rect( 0, hwy, hwx, py ) );
  cv::Mat q22( src, cv::Rect( px, 0, hwx, py ) );

  cv::Mat  q3( src, cv::Rect( hwx, hwy, px, py ) );
  cv::Mat q32( src, cv::Rect( 0, 0, px, py ) );

  // q3 is the biggest segment and might overlap with
  // everything else so it gets done last, we hold it in tmp
  cv::Mat tmpq3, tmpq1;
  q3.copyTo( tmpq3 );
  q1.copyTo( tmpq1 );

  q0.copyTo( q02 );
  q2.copyTo( q22 );
  tmpq3.copyTo( q32 );
  tmpq1.copyTo( q12 );
}

float sg::median( cv::Mat const & matrix )
{
  cv::Mat vec = matrix.reshape( 0, 1 );
  cv::sort( vec, vec, 0 );

  size_t half = vec.cols / 2;
  float y = vec.at<float>( 0, half + 1 );

  if( ( 2 * half ) == vec.cols )
    y = ( vec.at<float>( 0, half ) + y ) / 2.0f;

  return y;
}

//! Normalizes the GPU matrix in place
void sg::normalizeGPU( cv::gpu::GpuMat & mat, float max )
{
  double mi, ma;
  cv::gpu::minMax( mat, &mi, &ma );

	if( ma - mi == 0 )
		return;

  cv::gpu::subtract( mat, mi, mat );
  cv::gpu::multiply( mat, max / (ma - mi ), mat );
}

cv::Mat sg::padImageFFT( cv::Mat const & input, cv::Size const & paddedSize )
{
  cv::Mat result;
  const int m = paddedSize.height;
  const int n = paddedSize.width;

  cv::copyMakeBorder( input, result, 0, m - input.rows, 0, n - input.cols, cv::BORDER_REPLICATE );

  cv::Mat planes[] = { result, cv::Mat::zeros( result.size(), CV_32F ) };

  cv::merge( planes, 2, result );

  return result;
}

cv::Mat sg::padImageFFT( cv::Mat const & input, cv::Size const & kernelSize, cv::Size const & paddedSize )
{
  cv::Mat result;
  const int m = paddedSize.height;
  const int n = paddedSize.width;

  const int kernelX = kernelSize.width / 2;
  const int kernelY = kernelSize.height / 2;

  cv::copyMakeBorder( input, result, 0, m - input.rows, 0, n - input.cols, cv::BORDER_REPLICATE );

  // Handle wrap around cases
  cv::Mat      case3 = result.col( 0 );
  for( int i = 1; i <= kernelX; ++i )
  {
    cv::Mat dest = result.col( n - i );
    case3.copyTo( dest );
  }

  cv::Mat      case4 = result.row( 0 );
  for( int i = 1; i <= kernelY; ++i )
  {
    cv::Mat dest = result.row( m - i );
    case4.copyTo( dest );
  }

  cv::Mat planes[] = { result, cv::Mat::zeros( result.size(), CV_32F ) };

  cv::merge( planes, 2, result );

  return result;
}

cv::Mat sg::padKernelFFT( cv::Mat const & input, cv::Size const & paddedSize )
{
  cv::Mat result = cv::Mat::zeros( paddedSize, input.type() );

  // Get the 4 quadrants of the input kernel
  int hwx = input.cols / 2;
  int hwy = input.rows / 2;
  int px = hwx;
  int py = hwy;

  int fftW = paddedSize.width;
  int fftH = paddedSize.height;

  if( input.cols %2 != 0 )
    ++px;

  if( input.rows % 2 != 0 )
    ++py;

  cv::Mat  q0( input, cv::Rect( 0, 0, px, py ) );
  cv::Mat q02( result, cv::Rect( fftW - px, fftH - py, px, py ) );

  cv::Mat  q1( input, cv::Rect( px, 0, hwx, py ) );
  cv::Mat q12( result, cv::Rect( 0, fftH - py, hwx, py ) );

  cv::Mat  q2( input, cv::Rect( 0, py, px, hwy ) );
  cv::Mat q22( result, cv::Rect( fftW - px, 0, px, hwy ) );

  cv::Mat  q3( input, cv::Rect( px, py, hwx, hwy ) );
  cv::Mat q32( result, cv::Rect( 0, 0, hwx, hwy ) );

  q0.copyTo( q02 );
  q1.copyTo( q12 );
  q2.copyTo( q22 );
  q3.copyTo( q32 );

  // if our filter doesn't have real and complex components, make the complex part zeros
  if( result.channels() < 2 )
  {
    cv::Mat planes[] = { result, cv::Mat::zeros( result.size(), CV_32F ) };
    cv::merge( planes, 2, result );
  }

  return result;
}

cv::Size sg::getPaddedSize( cv::Size const & imageSize )
{
  // cvsize(width, height)
  return cv::Size( cv::getOptimalDFTSize( imageSize.width ),
                   cv::getOptimalDFTSize( imageSize.height ) );
}

cv::Size sg::getPaddedSize( cv::Size const & imageSize, cv::Size const & largestKernelSize )
{
  // cvsize(width, height)
  return cv::Size( cv::getOptimalDFTSize( imageSize.width + largestKernelSize.width - 1 ),
                   cv::getOptimalDFTSize( imageSize.height + largestKernelSize.height - 1 ) );
}

void sg::uncropFFTGPU( cv::gpu::GpuMat const & input, cv::gpu::GpuMat & output, std::vector<cv::gpu::GpuMat> & splitBuffer )
{
	cv::Size originalSize = input.size();
	cv::Size oldSize = output.size();

  if( input.channels() > 1 )
  {
		cv::gpu::split( input, splitBuffer );
    splitBuffer[0]( cv::Rect( 0, 0, oldSize.width, oldSize.height ) ).copyTo( output );
  }
	else
  	input( cv::Rect( 0, 0, oldSize.width, oldSize.height ) ).copyTo( output );

	cv::gpu::multiply( output, ( 1.0 / ( originalSize.width * originalSize.height ) ), output );
}

void sg::resizeInPlace( cv::Mat & input, cv::Size const & sz, double dx, double dy )
{
  cv::Mat temp;
  cv::resize( input, temp, sz, dx, dy );
  input = temp;
}

