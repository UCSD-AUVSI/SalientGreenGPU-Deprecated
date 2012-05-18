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

#include <SalientGreenGPU/Normalization/Normalization.H>
#include <SalientGreenGPU/Filters/Gaussian.H>
#include <SalientGreenGPU/Filters/LowPass.H>
#include <SalientGreenGPU/Filters/Utility.H>
#include <SalientGreenGPU/Cuda/dcAdjust.H>

void sg::maxNormalize( cv::gpu::GpuMat & input )
{
  float sum = 0;
  int num = 0;

	double mi, ma;
  cv::gpu::minMax( input, &mi, &ma );

  cv::gpu::subtract( input, mi, input );
  cv::gpu::multiply( input, 10.0f * ( ma - mi ), input );

  // calculate threshold
  cv::gpu::minMax( input, &mi, &ma );
  const float thresh = mi + ( ma - mi ) / 10.0;

	minMaxHelperWrapper( input, thresh, sum, num );

  // scale factor is (max-mean)^2
  if( num > 1 )
  {
    float factor = ma - sum / num;
		cv::gpu::multiply( input, factor * factor, input );
  }
  // single narrow peak
  else if( num == 1 )
		cv::gpu::multiply( input, ma * ma, input );
  // else no real peaks, leave alone
 
	normalizeGPU( input, 255 );
}


cv::Mat sg::maxNormalize( cv::Mat const & input )
{
  const int h = input.rows;
  const int w = input.cols;

  float sum = 0;
  int num = 0;

  double min, max;
  cv::minMaxLoc( input, &min, &max );

  // normalized to our new range
  cv::Mat result = ( input - min ) * ( 10.0f * ( max - min ) );
  auto iter = result.begin<float>();

  // calculate threshold
  cv::minMaxLoc( input, &min, &max );
  const float thresh = min + ( max - min ) / 10.0;

  // get mean of local maxima
  for( int j = 1; j < h - 1; ++j )
    for( int i = 1; i < w - 1; ++i )
    {
      const int index = i + w * j;
      const float val = iter[index];

      if( val >= thresh &&
          val >= iter[index - w] &&
          val >= iter[index + w] &&
          val >= iter[index - 1] &&
          val >= iter[index + 1] )
      {
        sum += val;
        ++num;
      }
    }

  // scale factor is (max-mean)^2
  if( num > 1 )
  {
    float factor = max - sum / num;
    result *= ( factor * factor );
  }
  // single narrow peak
  else if( num == 1 )
    result *= ( max * max );
  // else no real peaks, leave alone
  
	cv::normalize( result, result, 0, 255, cv::NORM_MINMAX );

  return result;
}

sg::NormalizeIterative::NormalizeIterative() : itsInitialized( false )
{ }

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

sg::NormalizeIterative::NormalizeIterative( cv::Size const & imageSize, cv::Size const & fftSize ) : itsInitialized( true ),
	itsImageSize( imageSize ), itsFFTSize( fftSize ),
	itsInhibition( fftSize, CV_32FC2 ), itsExcitation( fftSize, CV_32FC2 ),
	itsResultBuffer( fftSize, CV_32FC2 ), itsDFTBuffer( fftSize, CV_32FC2 )
{
	int width = imageSize.width;
	int height = imageSize.height;
  int size = ( width > height ) ? width : height;
  int minSize = ( width < height ) ? width : height;
  int maxhw = ( ( minSize / 2.0 - 1 ) > 0 ) ? ( minSize / 2.0 - 1.0 ) : 0;

  double inhibSig = size * inhSig * 0.01;
  double exhitSig = size * exhSig * 0.01;

  // create kernels
	cv::Mat inhib = gaussian2D( inhCoef / ( inhibSig * std::sqrt( 2 * M_PI ) ), inhibSig, maxhw );
	cv::Mat excit = gaussian2D( exhCoef / ( exhitSig * std::sqrt( 2 * M_PI ) ), exhitSig, maxhw );

	inhib = padKernelFFT( inhib, fftSize );
	excit = padKernelFFT( excit, fftSize );
			
	cv::gpu::Stream stream;

	stream.enqueueUpload( inhib, itsInhibition );
	stream.enqueueUpload( excit, itsExcitation );

	stream.waitForCompletion();

	cv::gpu::dft( itsInhibition, itsInhibition, fftSize );
	cv::gpu::dft( itsExcitation, itsExcitation, fftSize );

	subFC2Wrapper( itsExcitation, itsInhibition, itsExcitation );
}

void sg::NormalizeIterative::release()
{
	itsInhibition.release();
	itsExcitation.release();
	itsResultBuffer.release();
	itsDFTBuffer.release();
}


void sg::NormalizeIterative::operator()( cv::gpu::GpuMat const & input, cv::gpu::GpuMat & output, 
		std::vector<cv::gpu::GpuMat> & splitBuffer, int numIter )
{
	output.setTo( 0.0 );

  if( !itsInitialized )
    return;

	if( input.channels() > 1 )
	{
		cv::gpu::dft( input, itsResultBuffer, itsFFTSize );
	}
	else
	{
		// pad and take DFT
		itsResultBuffer.setTo( 0.0f );
		splitBuffer[1].setTo( 0.0f );
		cv::gpu::GpuMat merge[] = { input, splitBuffer[1] };
		cv::gpu::merge( merge, 2, itsResultBuffer );
		cv::gpu::dft( itsResultBuffer, itsResultBuffer, itsFFTSize );
	}

	cv::gpu::GpuMat const & filter = itsExcitation;

  for( int i = 0; i < numIter; ++i )
  {
    double min, max;

		cv::gpu::mulSpectrums( itsResultBuffer, filter, itsDFTBuffer, cv::DFT_COMPLEX_OUTPUT );
	
		// Add the filter result to our current output
		addFC2Wrapper( itsResultBuffer, itsDFTBuffer, itsResultBuffer );

		// Get min and max from DFT result - we need to do this in the spatial domain
		// and crop only the image sized portion out
		cv::gpu::dft( itsDFTBuffer, itsDFTBuffer, itsResultBuffer.size(), cv::DFT_INVERSE );
		uncropFFTGPU( itsDFTBuffer, output, splitBuffer );
		cv::gpu::minMax( output, &min, &max );
		
    float const globalInhib = -0.01f * inhibCoef * max;	

		// now we'll add the global inhibition and clamp to zero
		// we also zero the complex component just in case
		addAndZeroFC2Wrapper( itsDFTBuffer, globalInhib );
		maxFC2Wrapper( itsDFTBuffer, 0.0 );
		
		//uncropFFTGPU( itsDFTBuffer, itsInhibition, splitBuffer );
		//cv::Mat temp( itsInhibition.size(), CV_32FC1 );
		//temp = itsInhibition;
		//cv::normalize(temp,temp,0,1,cv::NORM_MINMAX);
		//cv::imshow("sdf",temp);cv::waitKey(0);

		// prepare for next iteration by taking dft
		if( i < numIter - 1 )
			cv::gpu::dft( itsDFTBuffer, itsResultBuffer, itsFFTSize );
  }
	
	// Discard complex component and crop to just the image
	if( numIter < 1 )
		uncropFFTGPU( input, output, splitBuffer );
	else
		uncropFFTGPU( itsDFTBuffer, output, splitBuffer );
    
	//cv::Mat temp2( output.size(), output.type() );
	//temp2 = output;
	///cv::normalize(temp2,temp2,0,1,cv::NORM_MINMAX);
	//cv::imshow("sdf",temp2);cv::waitKey(0);
	//cv::gpu::absdiff( output, 0.0, output );
	normalizeGPU( output, 255.0f );
}

/*! Adapted from the iLab Neuromorphic Robotics Toolkit (NRT)
 * Copyright 2010 by the University of Southern California (USC) and
 * the iLab at USC.
 *
 * See http://ilab.usc.edu for information about this project
 * and licensing information */
void sg::attenuateBorders( cv::Mat & mat, int size )
{
  auto dims = mat.size();

  if (size * 2 > dims.width) size = dims.width / 2;
  if (size * 2 > dims.height) size = dims.height / 2;
  if (size < 1) return;  // forget it

  float increment = 1.0 / (float)(size + 1);
  // top lines:
  float coeff = increment;
  auto aptr = mat.begin<float>();
  for (int y = 0; y < size; ++ y)
  {
    for (int x = 0; x < dims.width; ++ x)
    {
      *aptr = (float)( (*aptr) * coeff );
      ++aptr;
    }
    coeff += increment;
  }
  // normal lines: start again from beginning to attenuate corners twice:
  aptr = mat.begin<float>();
  for (int y = 0; y < dims.height; ++ y)
  {
    coeff = increment;
    for (int x = 0; x < size; ++ x)
    {
      *(aptr + dims.width - 1 - x * 2) =
        (float)(*(aptr + dims.width - 1 - x * 2) * coeff);

      *aptr = (float)( (*aptr) * coeff );
      ++aptr;
      coeff += increment;
    }
    aptr += dims.width - size;
  }
  // bottom lines
  aptr = mat.begin<float>() + (dims.height - size) * dims.width;
  coeff = increment * (float)size;
  for (int y = dims.height - size; y < dims.height; ++ y)
  {
    for (int x = 0; x < dims.width; ++ x)
    {
      *aptr = (float)( (*aptr) * coeff );
      ++aptr;
    }
    coeff -= increment;
  }
}
