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

#include <SalientGreenGPU/SalientGreenGPU.H>
#include <SalientGreenGPU/Filters/LowPass.H>
#include <SalientGreenGPU/Filters/Utility.H>
#include <SalientGreenGPU/Cuda/dcAdjust.H>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/gpu/gpu.hpp>

#include <iostream>

void sg::SalientGreenGPU::Results::show()
{
  cv::Mat temp;
  cv::resize( lResponse, temp, labSaliency.size() );
  cv::imshow( "lResponse", temp / 255.0f );
  cv::resize( aResponse, temp, labSaliency.size() );
  cv::imshow( "aResponse", temp / 255.0f );
  cv::resize( bResponse, temp, labSaliency.size() );
  cv::imshow( "bResponse", temp / 255.0f );
  cv::resize( oResponse, temp, labSaliency.size() );
  cv::imshow( "oResponse", temp / 255.0f );

  cv::imshow( "labSaliency", labSaliency / 255.0f );
}

void sg::SalientGreenGPU::Results::save(
  std::string const & prefix, std::string const & filetype )
{
  cv::Mat temp;
  cv::resize( lResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "lResponse." + filetype, temp );
  cv::resize( aResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "aResponse." + filetype, temp );
  cv::resize( bResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "bResponse." + filetype, temp );
  cv::resize( oResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "oResponse." + filetype, temp );

  cv::imwrite( prefix + "saliencyLAB." + filetype, labSaliency );
}

sg::SalientGreenGPU::SalientGreenGPU() :
  itsInputImage()
{
}

void sg::SalientGreenGPU::release()
{
	for( auto & i : itsLabResponse )
    i.release();

	itsOResponse.release();

	for( auto & i : itsLabFFT )
    i.release();
	
	itsGrayscaleFFT.release();
  itsOBuffer.release();

	for( auto & i : itsSplitBufferFrequency )
    i.release();
	
	for( auto & i : itsSplitBufferSpatial )
    i.release();

  itsSaliencyLab.release();
  itsGpuBuffer.release();

  itsGpuBufferSpatial.release();

	for( auto & i : itsGpuBufferFrequency )
    i.release();
  
	itsDoGBank.release();
  itsDoGBankFrequency.release();
  itsLogGabor.release();
  itsNormalizeIterSpatial.release();
  itsNormalizeIterFrequency.release();
}

auto sg::SalientGreenGPU::computeSaliencyGPU( cv::Mat const & image,
    labWeights const * lw ) -> Results
{
  // NOTES: if lw && sw == NULL, this won't work
  auto prevSize = itsInputImage.size();
  itsInputImage = image;

  // convert to float, leave 0 to 255 range
  if( image.type() != CV_32FC3 )
    itsInputImage.convertTo( itsInputImage, CV_32FC3 );

  // Get the input image on the GPU
  // itsInputImageGPU.upload( itsInputImage ); // LEWIS TEMP: remove soon

  // do we need to generate new filters? (we will if this is a different size!)
  if( prevSize != itsInputImage.size() )
  {
		release();
    prepareFiltersGPU();
    allocateGPUBuffers( );//lw != nullptr ); // LEWIS
  }

  // Compute saliency
  Results res( itsInputImage.size() );
  cv::Mat saliency;

  cv::Mat grayscale;
  cv::Mat grayscaleFFT;

  std::vector<cv::Mat> lab( 3 );
  std::vector<cv::Mat> labFFT( 3 );

  // get grayscale
  // cv::cvtColor( itsInputImage, grayscale, CV_BGR2GRAY ); // LEWIS: removed

  // get lab
	{
		cv::cvtColor( itsInputImage, saliency, CV_BGR2Lab );
		cv::split( saliency, lab );
	}

  // Prepare to take FFT
  std::cout << "Preparing images\n";

  // grayscaleFFT = padImageFFT( grayscale, itsFFTSizeFrequency ); // LEWIS: removed
 
	// Lab FFTs
  {
    labFFT[0] = padImageFFT( lab[0], itsMaxFilterSize, itsFFTSizeSpatial );
    labFFT[1] = padImageFFT( lab[1], itsMaxFilterSize, itsFFTSizeSpatial );
    labFFT[2] = padImageFFT( lab[2], itsMaxFilterSize, itsFFTSizeSpatial );
  }

  // Upload data to the GPU
  std::cout << "Uploading images to GPU\n";

  cv::gpu::Stream stream;

  // stream.enqueueUpload( grayscaleFFT, itsGrayscaleFFT ); // LEWIS: removed
	for( size_t i = 0; i < 3; ++i )
  {
    stream.enqueueUpload( labFFT[i], itsLabFFT[i] );
  }

  stream.waitForCompletion();

  // FFT images
  std::cout << "Taking image DFTs\n";
  // cv::gpu::dft( itsGrayscaleFFT, itsGrayscaleFFT, itsGrayscaleFFT.size() ); // LEWIS: removed

	for( auto & i : itsLabFFT )
      cv::gpu::dft( i, i, i.size() );

  itsGrayscaleFFT = itsLabFFT[0]; // LEWIS: use luminosity for gabor

  // Compute saliency!
  // gabor
  {
    std::cout << "Processing orientation maps\n";
    doGaborGPU( itsGrayscaleFFT, itsOBuffer );
  }

  // lab color
	{
    std::cout << "Calculating LAB saliency\n";
    std::cout << "L channel...\n";
    doDoGGPU( itsLabFFT[0], itsLabFFT[0] );
    std::cout << "A channel...\n";
    doDoGGPU( itsLabFFT[1], itsLabFFT[1] );
    std::cout << "B channel...\n";
    doDoGGPU( itsLabFFT[2], itsLabFFT[2] );
  }

  std::cout << "Normalizing\n";
  // gabor
  {
    itsNormalizeIterFrequency( itsOBuffer, itsOResponse, itsSplitBufferFrequency, 1 );
  }

  // lab
	{
    for( size_t i = 0; i < 3; ++i )
      itsNormalizeIterSpatial( itsLabFFT[i], itsLabResponse[i], itsSplitBufferSpatial, 2 );
  }

  // Accumulate maps on GPU
  std::cout << "Accumulating maps\n";

  // lab
	{
    itsSaliencyLab.setTo( 0.0 );
    cv::gpu::multiply( itsLabResponse[0], lw->l, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

    cv::gpu::multiply( itsLabResponse[1], lw->a, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

    cv::gpu::multiply( itsLabResponse[2], lw->b, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

    cv::gpu::multiply( itsOResponse, lw->o, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

		cv::Mat temp( itsSaliencyLab.size(), itsSaliencyLab.type() );
		// temp = itsSaliencyLab;
    itsSaliencyLab.download(temp); // LEWIS
		// itsLabFFT[0] = padImageFFT( temp, itsMaxFilterSize, itsFFTSizeSpatial );
    itsLabFFT[0].upload(padImageFFT( temp, itsMaxFilterSize, itsFFTSizeSpatial )); // LEWIS
    
    itsNormalizeIterSpatial( itsLabFFT[0], itsSaliencyLab, itsSplitBufferSpatial, 0 );
  }

  // Download everything
  {
    stream.enqueueDownload( itsLabResponse[0], res.lResponse );
    stream.enqueueDownload( itsLabResponse[1], res.aResponse );
    stream.enqueueDownload( itsLabResponse[2], res.bResponse );
  }

	stream.enqueueDownload( itsSaliencyLab, res.labSaliency );
  stream.enqueueDownload( itsOResponse, res.oResponse );
	// LEWIS: you only need to download saliency here when you want this
	// going as fast as possible, the others are just so you can look at them
	// and debug things for now

  stream.waitForCompletion();

  return res;
}

void sg::SalientGreenGPU::allocateGPUBuffers()
{
  std::cout << "Allocating memory on the GPU.\n";

  // We force a release of all prior stuff just in case OpenCV isn't
  // deallocating things when we want more space on the GPU

	// LAB response is same size as input, one channel floating point
	for( auto & i : itsLabResponse )
		i = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );

  // O and S response is same size as input, one channel floating point
  itsOResponse = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );

	// FFT lab are sized according to fftsizespatial, two channels
	for( auto & i : itsLabFFT )
		i = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC2 );

  // FFT for grayscale is sized according to fftsizeFrequency, two channels
  itsGrayscaleFFT = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );
  itsOBuffer = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );

  // Buffer used for splitting images
  itsSplitBufferFrequency.resize( 2 );
	for( auto & i : itsSplitBufferFrequency )
    i = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC1 );

  itsSplitBufferSpatial.resize( 2 );
	for( auto & i : itsSplitBufferSpatial )
    i = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC1 );

  itsSaliencyLab = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  itsGpuBuffer = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );

  itsGpuBufferSpatial = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC2 );
	for( auto & i : itsGpuBufferFrequency )
    i = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );
}

void sg::SalientGreenGPU::prepareFiltersGPU()
{
  std::cout << "Preparing all filters.\n";

  // Figure out FFT size for spatial based
  int maxhw = static_cast<int>( std::min( itsInputImage.rows / 2.0 - 1, itsInputImage.cols / 2.0 - 1 ) );
  int largest = DoGCenterSurround::largestFilterSize( maxhw );

  itsMaxFilterSize = cv::Size( largest, largest );

  itsFFTSizeSpatial = getPaddedSize( itsInputImage.size(),
                                     itsMaxFilterSize );

  // FIgure out FFT size for frequency based kernels
  itsFFTSizeFrequency = getPaddedSize( itsInputImage.size() );

  // Create DoG filters
  itsDoGBank = DoGCenterSurround( maxhw, itsFFTSizeSpatial );
  itsDoGBankFrequency = DoGCenterSurround( maxhw, itsFFTSizeFrequency );

  // Create Gabor filters
  cv::Mat lp = lowpass( itsFFTSizeFrequency.width, itsFFTSizeFrequency.height, 0.4, 10 );
  itsLogGabor = LogGabor( itsFFTSizeFrequency.width, itsFFTSizeFrequency.height, lp );//,
			//6, 6.0, 3.0, 0.55, 2.0 );
  itsLogGabor.addFilters();

  // Create iterative normalizer
  itsNormalizeIterSpatial = NormalizeIterative( itsInputImage.size(), itsFFTSizeSpatial );
  itsNormalizeIterFrequency = NormalizeIterative( itsInputImage.size(), itsFFTSizeFrequency );

  std::cout << "Filter creation finished.\n";
}

void sg::SalientGreenGPU::doGaborGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & gabor )
{
  itsLogGabor.getEdgeResponses( fftImage, itsGpuBufferFrequency[0], itsSplitBufferFrequency, itsDoGBankFrequency );

  cv::gpu::dft( itsGpuBufferFrequency[0], gabor, gabor.size(), cv::DFT_INVERSE );
  absFC2Wrapper( gabor );
}

void sg::SalientGreenGPU::doDoGGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & output )
{
  itsDoGBank( fftImage, itsGpuBufferSpatial );

  cv::gpu::dft( itsGpuBufferSpatial, output, output.size(), cv::DFT_INVERSE );
  absFC2Wrapper( output );
}
