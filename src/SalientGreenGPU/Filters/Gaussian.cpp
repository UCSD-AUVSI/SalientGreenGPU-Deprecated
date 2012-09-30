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

#include <SalientGreenGPU/Filters/Gaussian.H>
#include <SalientGreenGPU/Filters/Utility.H>
#include <SalientGreenGPU/Cuda/dcAdjust.H>


cv::Mat sg::gaussian( double peak, double sigma, int maxhw, double threshPercent )
{
  int hw = static_cast<int>( sigma * std::sqrt( -2.0 * std::log( threshPercent / 100.0 ) ) );

  // trim hw values out of range
  if( maxhw > 0 && hw > maxhw )
    hw = maxhw;

  int fw = 2 * hw + 1;

  cv::Mat result( 1, fw, CV_32FC1 );

  if( peak == 0 )
    peak = 1.0 / ( sigma * std::sqrt( 2 * M_PI ) );

	float sigma22 = -0.5f / ( sigma * sigma );

	for( int i = 1; i <= hw; ++i )
	{
		float value = peak * std::exp( static_cast<float>( i * i ) * sigma22 );
		result.at<float>( 0, hw + i ) = value;
		result.at<float>( 0, hw - i ) = value;
	}

	result.at<float>( 0, hw ) = peak;
	result /= cv::sum( result )[0];

  return result;
}

cv::Mat sg::gaussian2D( double peak, double sigma, int maxhw, double threshPercent )
{
	cv::Mat gauss1d = gaussian( peak, sigma, maxhw, threshPercent );

	cv::Mat result( gauss1d.cols, gauss1d.cols, gauss1d.type() );

	for( int i = 0; i < result.rows; ++i )
		for( int j = 0; j < result.cols; ++j )
			result.at<float>( i, j ) = gauss1d.at<float>( 0, i ) * gauss1d.at<float>( 0, j );

	result /= cv::sum( result )[0];

	return result;
}

sg::DoGCenterSurround::DoGCenterSurround() : itsIsValid( false )
{ }

sg::DoGCenterSurround::DoGCenterSurround( int maxhw, cv::Size const & size,
    double initSigma, double mult, int scales, double scaleFactor ) :
  itsIsValid( true ), itsBuffer( size, CV_32FC2 )
{
	// Allocate GPU filters
  itsFilters.resize( scales );
	//for( cv::gpu::GpuMat filt : itsFilters )
	for (int i = 0; i < itsFilters.size(); i++)
		itsFilters.at(i) = cv::gpu::GpuMat( size, CV_32FC2 );

	// Create filters on CPU then upload them
  double curSigma = initSigma;

	cv::Mat current[4]; // scales = 4 need to keep array of them all so we can upload in a stream
	cv::gpu::Stream stream;

  for( int i = 0; i < scales; ++i )
  {
    const double narrowS2 = curSigma * curSigma;
    const double wideS2 = narrowS2 * mult * mult;

    cv::Mat narrow = gaussian2D( 1.0 / narrowS2, curSigma, maxhw );
    cv::Mat wide   = gaussian2D( 1.0 / wideS2,   curSigma * mult, maxhw );

		narrow = padKernelFFT( narrow, size );
		wide   = padKernelFFT( wide,   size );

		cv::dft( narrow, narrow );
		cv::dft( wide, wide );

    current[i] = narrow - wide; 
    curSigma *= scaleFactor;

		// Upload to GPU
		stream.enqueueUpload( current[i], itsFilters[i] );
  }

	stream.waitForCompletion();
}

void sg::DoGCenterSurround::release()
{
	for (int i = 0; i < itsFilters.size(); i++)
		itsFilters.at(i).release();
	itsBuffer.release();
}

void sg::DoGCenterSurround::operator()( cv::gpu::GpuMat const & input, cv::gpu::GpuMat & output )
{
	if( !itsIsValid )
		return;

	// Zero the output 
	output.setTo( 0.0f );

	for (int i = 0; i < itsFilters.size(); i++)
	{
		cv::gpu::mulSpectrums( input, itsFilters.at(i), itsBuffer, cv::DFT_COMPLEX_OUTPUT );
		addFC2Wrapper( output, itsBuffer, output );
	}

	mulValueFC2Wrapper( output, 1.0f / itsFilters.size(), output );
}

int sg::DoGCenterSurround::largestFilterSize( int maxhw, double initSigma, double mult, int scales, double scaleFactor )
{
	for( int i = 0; i < scales; ++i )
		initSigma *= scaleFactor;

	initSigma *= mult;

  int hw = static_cast<int>( initSigma * std::sqrt( -2.0 * std::log( 1.0 / 100.0 ) ) );

  // trim hw values out of range
  if( maxhw > 0 && hw > maxhw )
    hw = maxhw;

  int fw = 2 * hw + 1;

	return fw;
}
