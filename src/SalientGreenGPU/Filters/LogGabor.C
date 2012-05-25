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
 * <http://www.gnu.org/licenses/>
 *
 * Portions of this file are based on Matlab softare licensed
 * under the MIT license by
 * Peter Kovesi
 * Centre for Exploration Targeting
 * School of Earth and Environment
 * The University of Western Australia
 *
 * See <www.csse.uwa.edu.au/~pk/research/matlabfns/license.html>
 * for more details */

#include <SalientGreenGPU/Filters/LogGabor.H>
#include <SalientGreenGPU/Filters/Utility.H>
#include <SalientGreenGPU/Cuda/dcAdjust.H>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace
{
  bool isSame( cv::gpu::GpuMat const & a, cv::gpu::GpuMat const & b )
  {
    if( a.size() == b.size() &&
        a.depth() == b.depth() &&
        a.data == b.data )
      return true;
    else
      return false;
  }
}

sg::LogGabor::LogGabor() : itsValid( false )
{
}

sg::LogGabor::LogGabor( int width, int height, cv::Mat const & lowpass,
                        int numScales, double minWaveLength,
                        double mult, double sigmaOnf ) :
  itsValid( false ), itsWidth( width ), itsHeight( height ),
  itsMult( mult ),
  itsSine( height, width, CV_32FC1 ), itsCosine( height, width, CV_32FC1 )
{
  itsGaborScales.resize( numScales );

  // Create radius, sine, and cosine matrices
  cv::Mat radius( height, width, CV_32FC1 );

  std::vector<float> xData, yData;
  xData.resize( width ); yData.resize( height );

  double widthVal = ( width % 2 ) ? width - 1 : width;
  double heightVal = ( height % 2 ) ? height - 1 : height;

  double val = -widthVal / 2.0;
	for( auto & entry : xData )
  {
    entry = val / widthVal;
    val += 1.0;
  }

  val = -heightVal / 2.0;
	for( auto & entry : yData )
  {
    entry = val / heightVal;
    val += 1.0;
  }

  // fill out values
  for( int i = 0; i < height; ++i )
    for( int j = 0; j < width; ++j )
    {
      radius.at<float>( i, j ) = std::sqrt( xData[j] * xData[j] + yData[i] * yData[i] );
      float temp = std::atan2( -yData[i], xData[j] );
      itsCosine.at<float>( i, j ) = std::cos( temp );
      itsSine.at<float>( i, j ) = std::sin( temp );
    }

  ifftshift( radius );
  ifftshift( itsSine );
  ifftshift( itsCosine );
  radius.at<float>( 0, 0 ) = 1.0f;

  // Fill out our log gabors for each scale
  double sigmaScale = 2 * std::log( sigmaOnf ) * std::log( sigmaOnf );
  for( int i = 0; i < numScales; ++i )
  {
    const float wavelength = minWaveLength * std::pow( mult, i );
    const float fo = 1.0f / wavelength;

    cv::Mat filter; // temp value
    cv::log( radius / fo, filter );
    cv::multiply( filter, filter, filter );
    cv::exp( -filter / sigmaScale, filter );

    cv::multiply( filter, lowpass, itsGaborScales[i] );
    itsGaborScales[i].at<float>( 0, 0 ) = 0;
  }
}

void sg::LogGabor::release()
{
	for( auto & i : itsFilters )
		for( auto & j : i )
			j.release();

	itsBufferReal.release();
	itsBufferImag.release();

	for( auto & i : itsFFTBuffer )
		i.release();
}


void sg::LogGabor::addFilters( int numOrientations )
{
  // Keep things locally for a while so we can upload in a stream
  std::vector<cv::Mat> filters( numOrientations * itsGaborScales.size() );
  cv::gpu::Stream stream;

	itsFilters.resize( numOrientations );
	for( auto & o : itsFilters )
		o.resize( itsGaborScales.size() );

	// Allocate GPU filters and buffers
	for( auto & o : itsFilters )
		for( auto & s : o )
			s = cv::gpu::GpuMat( itsHeight, itsWidth, CV_32FC2 );

	for( auto & i : itsFFTBuffer )
		i = cv::gpu::GpuMat( itsHeight, itsWidth, CV_32FC2 );
	
  // for each orientation...
  for( int o = 0; o < numOrientations; ++o )
  {
    const float orientation = o * M_PI / numOrientations;

    // spread function ranges 0 to 1
    cv::Mat spread( itsHeight, itsWidth, CV_32FC1 );

    const float cosOri = std::cos( orientation );
    const float sinOri = std::sin( orientation );

    auto sineIter = itsSine.begin<float>();
    auto cosineIter = itsCosine.begin<float>();

    for( auto i = spread.begin<float>(); i != spread.end<float>(); ++i, ++sineIter, ++cosineIter )
    {
      const float ds = *sineIter * cosOri - *cosineIter * sinOri;
      const float dc = *cosineIter * cosOri + *sineIter * sinOri;
      float dt = std::abs( std::atan2( ds, dc ) );
      dt = std::min( dt * numOrientations / 2.0f, static_cast<float>( M_PI ) );

      *i = ( std::cos( dt ) + 1.0f ) / 2.0f;
    }

    // loop over scales
    for( size_t s = 0; s < itsGaborScales.size(); ++s )
    {
      cv::Mat temp = itsGaborScales[s].mul( spread );
      cv::Mat output( itsHeight, itsWidth, CV_32FC2 );

      // note: technically our complex component should be
      // zero valued, but when we actually apply this the filter
      // in getFilterResponse we'll use a normal element wise
      // multiplication which has the same effect as a proper
      // complex * complex multiplication when the complex part
      // is zero
      cv::Mat out[] = { temp, temp };
      cv::merge( out, 2, output );

      // Upload to GPU
      filters.push_back( output );
      stream.enqueueUpload( filters.back(), itsFilters[o][s] );
    }
  }

  itsValid = true;
}

void sg::LogGabor::getEdgeResponses( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & edges,
		std::vector<cv::gpu::GpuMat> & splitBuffer, DoGCenterSurround & dog )
{
	if( !itsValid )
	{
		std::cerr << "This LogGabor bank is not initialized!\n";
		return;
	}

	edges.setTo( 0.0f );
  
  // outer loop for orientations
  for( size_t o = 0; o < itsFilters.size(); ++o )
  {
    // inner loop over scales
    for( size_t s = 0; s < itsGaborScales.size(); ++s )
    {
      // see note in addFilters about why this is a normal multiplication
      // and not a spectrum multiply
			mulFC2Wrapper( fftImage, itsFilters[o][s], itsFFTBuffer[0] );
			cv::gpu::dft( itsFFTBuffer[0], itsFFTBuffer[1], itsFFTBuffer[0].size(), cv::DFT_INVERSE );

      // Get magnitude
      cv::gpu::magnitude( itsFFTBuffer[1], splitBuffer[0] );
      
			// Compute edge response
			// 	the edge response is a DoG applied to the magnitude
			// 	so we'll pad the magnitude with zero valued complex
			// 	take the DFT and then pass it through the DoG processing chain
			splitBuffer[1].setTo( 0.0f );
			cv::gpu::GpuMat merge[] = { splitBuffer[0], splitBuffer[1] };
			cv::gpu::merge( merge, 2, itsFFTBuffer[0] );
			cv::gpu::dft( itsFFTBuffer[0], itsFFTBuffer[0], itsFFTBuffer[0].size() );
			dog( itsFFTBuffer[0], itsFFTBuffer[2] );
			
			// accumulate result into edges
			addFC2Wrapper( edges, itsFFTBuffer[2], edges );
    } // end scales
  } // end orientations
}
