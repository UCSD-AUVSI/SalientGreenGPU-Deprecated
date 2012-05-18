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
  cv::resize( lResponse, temp, rgbSaliency.size() );
  cv::imshow( "lResponse", temp / 255.0f );
  cv::resize( aResponse, temp, rgbSaliency.size() );
  cv::imshow( "aResponse", temp / 255.0f );
  cv::resize( bResponse, temp, rgbSaliency.size() );
  cv::imshow( "bResponse", temp / 255.0f );
  cv::resize( iResponse, temp, rgbSaliency.size() );
  cv::imshow( "iResponse", temp / 255.0f );
  cv::resize( rgResponse, temp, rgbSaliency.size() );
  cv::imshow( "rgResponse", temp / 255.0f );
  cv::resize( byResponse, temp, rgbSaliency.size() );
  cv::imshow( "byResponse", temp / 255.0f );
  cv::resize( oResponse, temp, rgbSaliency.size() );
  cv::imshow( "oResponse", temp / 255.0f );
  cv::resize( sResponse, temp, rgbSaliency.size() );
  cv::imshow( "sResponse", temp / 255.0f );

  cv::imshow( "rgbSaliency", rgbSaliency / 255.0f );
  cv::imshow( "labSaliency", labSaliency / 255.0f );
  cv::imshow( "symSaliency", symSaliency / 255.0f );
}

void sg::SalientGreenGPU::Results::save(
  std::string const & prefix, std::string const & filetype )
{
  cv::Mat temp;
  cv::resize( lResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "lResponse." + filetype, temp );
  cv::resize( aResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "aResponse." + filetype, temp );
  cv::resize( bResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "bResponse." + filetype, temp );
  cv::resize( iResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "iResponse." + filetype, temp );
  cv::resize( rgResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "rgResponse." + filetype, temp );
  cv::resize( byResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "byResponse." + filetype, temp );
  cv::resize( oResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "oResponse." + filetype, temp );
  cv::resize( sResponse, temp, rgbSaliency.size() );
  cv::imwrite( prefix + "sResponse." + filetype, temp );

  cv::imwrite( prefix + "saliencyRGB." + filetype, rgbSaliency );
  cv::imwrite( prefix + "saliencyLAB." + filetype, labSaliency );
  cv::imwrite( prefix + "saliencySYM." + filetype, symSaliency );
}

sg::SalientGreenGPU::SalientGreenGPU() :
  itsInputImage()
{
}

void sg::SalientGreenGPU::release()
{
	for( auto & i : itsLabResponse )
    i.release();

	for( auto & i : itsBgrResponse )
    i.release();
  
	itsOResponse.release();
  itsSResponse.release();

	for( auto & i : itsLabFFT )
    i.release();
	
	for( auto & i : itsBgrFFT )
    i.release();
  
	itsGrayscaleFFT.release();
  itsOBuffer.release();
  itsSBuffer.release();

	for( auto & i : itsSplitBufferFrequency )
    i.release();
	
	for( auto & i : itsSplitBufferSpatial )
    i.release();

  itsSaliencyBgr.release();
  itsSaliencyLab.release();
  itsSaliencySym.release();
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
    rgbWeights const * rw,
    labWeights const * lw,
    symWeights const * sw ) -> Results
{
  // NOTES: if lw && sw == NULL, this won't work
  auto prevSize = itsInputImage.size();
  itsInputImage = image;

  // convert to float, leave 0 to 255 range
  if( image.type() != CV_32FC3 )
    itsInputImage.convertTo( itsInputImage, CV_32F );

  // Get the input image on the GPU
  itsInputImageGPU.upload( itsInputImage );

  // do we need to generate new filters? (we will if this is a different size!)
  if( prevSize != itsInputImage.size() )
  {
		release();
    prepareFiltersGPU();
    allocateGPUBuffers( rw != nullptr, lw != nullptr, sw != nullptr );
  }

  // Compute saliency
  Results res( itsInputImage.size() );
  cv::Mat saliency;

  cv::Mat grayscale;
  cv::Mat grayscaleFFT;

  std::vector<cv::Mat> lab( 3 );
  std::vector<cv::Mat> bgr( 3 );
  std::vector<cv::Mat> bgrFFT( 3 );
  std::vector<cv::Mat> labFFT( 3 );

  // get grayscale
  // cv::cvtColor( itsInputImage, grayscale, CV_BGR2GRAY ); // LEWIS: removed

  // get lab
  if( lw || sw )
  {
    cv::cvtColor( itsInputImage, saliency, CV_BGR2Lab );
    cv::split( saliency, lab );
  }

  // get rgb
  if( rw )
  {
    cv::split( itsInputImage, bgr );
    bgrFFT[0] = ( bgr[0] + bgr[1] + bgr[2] ) / 3.0f; // intensity
    bgrFFT[1] = ( bgr[2] - ( bgr[0] + bgr[1] ) / 2.0f ); // RG
    bgrFFT[2] = ( bgr[0] - ( bgr[1] + bgr[2] ) / 2.0f ); // BY
  }

  // Prepare to take FFT
  std::cout << "Preparing images\n";

  // grayscaleFFT = padImageFFT( grayscale, itsFFTSizeFrequency ); // LEWIS: removed

  if( lw || sw )
  {
    labFFT[0] = padImageFFT( lab[0], itsMaxFilterSize, itsFFTSizeSpatial );
    labFFT[1] = padImageFFT( lab[1], itsMaxFilterSize, itsFFTSizeSpatial );
    labFFT[2] = padImageFFT( lab[2], itsMaxFilterSize, itsFFTSizeSpatial );
  }

  if( rw )
  {
    bgrFFT[0] = padImageFFT( bgrFFT[0], itsMaxFilterSize, itsFFTSizeSpatial );
    bgrFFT[1] = padImageFFT( bgrFFT[1], itsMaxFilterSize, itsFFTSizeSpatial );
    bgrFFT[2] = padImageFFT( bgrFFT[2], itsMaxFilterSize, itsFFTSizeSpatial );
  }

  // Upload data to the GPU
  std::cout << "Uploading images to GPU\n";

  cv::gpu::Stream stream;

  // stream.enqueueUpload( grayscaleFFT, itsGrayscaleFFT ); // LEWIS: removed
  for( size_t i = 0; i < 3; ++i )
  {
    stream.enqueueUpload( labFFT[i], itsLabFFT[i] );
    stream.enqueueUpload( bgrFFT[i], itsBgrFFT[i] );
  }

  stream.waitForCompletion();

  // FFT images
  std::cout << "Taking image DFTs\n";
  // cv::gpu::dft( itsGrayscaleFFT, itsGrayscaleFFT, itsGrayscaleFFT.size() ); // LEWIS: removed

  if( lw || sw )
for( auto & i : itsLabFFT )
      cv::gpu::dft( i, i, i.size() );

  if( rw )
for( auto & i : itsBgrFFT )
      cv::gpu::dft( i, i, i.size() );

  itsGrayscaleFFT = itsLabFFT[0]; // LEWIS: use luminosity for gabor

  // Compute saliency!
  // gabor and symmetry
  {
    std::cout << "Processing orientation and symmetry maps\n";
    doSymmetryAndGaborGPU( itsGrayscaleFFT, itsSBuffer, itsOBuffer );
  }

  if( rw )
  {
    std::cout << "Calculating RGB saliency\n";
    std::cout << "Intensity channel...\n";
    doDoGGPU( itsBgrFFT[0], itsBgrFFT[0] );
    std::cout << "RG channel...\n";
    doDoGGPU( itsBgrFFT[1], itsBgrFFT[1] );
    std::cout << "BY channel...\n";
    doDoGGPU( itsBgrFFT[2], itsBgrFFT[2] );
  }

  if( lw || sw )
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

  if( rw )
  {
    for( size_t i = 0; i < 3; ++i )
      itsNormalizeIterSpatial( itsBgrFFT[i], itsBgrResponse[i], itsSplitBufferSpatial, 2 );
  }

  if( lw || sw )
  {
    for( size_t i = 0; i < 3; ++i )
      itsNormalizeIterSpatial( itsLabFFT[i], itsLabResponse[i], itsSplitBufferSpatial, 2 );
  }

  if( sw )
  {
    itsNormalizeIterFrequency( itsSBuffer, itsSResponse, itsSplitBufferFrequency, 1 );
  }

  // Accumulate maps on GPU
  std::cout << "Accumulating maps\n";

  if( rw )
  {
    itsSaliencyBgr.setTo( 0.0 );
    cv::gpu::multiply( itsBgrResponse[0], rw->i, itsGpuBuffer );

    cv::gpu::add( itsSaliencyBgr, itsGpuBuffer, itsSaliencyBgr );

    cv::gpu::multiply( itsBgrResponse[1], rw->rg, itsGpuBuffer );
    cv::gpu::add( itsSaliencyBgr, itsGpuBuffer, itsSaliencyBgr );

    cv::gpu::multiply( itsBgrResponse[2], rw->by, itsGpuBuffer );
    cv::gpu::add( itsSaliencyBgr, itsGpuBuffer, itsSaliencyBgr );

    cv::gpu::multiply( itsOResponse, rw->o, itsGpuBuffer );
    cv::gpu::add( itsSaliencyBgr, itsGpuBuffer, itsSaliencyBgr );
    
		cv::Mat temp( itsSaliencySym.size(), itsSaliencySym.type() );
		temp = itsSaliencyBgr;
		itsLabFFT[0] = padImageFFT( temp, itsMaxFilterSize, itsFFTSizeSpatial );

    itsNormalizeIterSpatial( itsLabFFT[0], itsSaliencyBgr, itsSplitBufferSpatial, 0 );
    //maxNormalize( itsSaliencyBgr );
  }

  if( lw )
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

		cv::Mat temp( itsSaliencySym.size(), itsSaliencySym.type() );
		temp = itsSaliencyLab;
		itsLabFFT[0] = padImageFFT( temp, itsMaxFilterSize, itsFFTSizeSpatial );

    itsNormalizeIterSpatial( itsLabFFT[0], itsSaliencyLab, itsSplitBufferSpatial, 0 );
    //maxNormalize( itsSaliencyLab );
  }

  if( sw )
  {
    itsSaliencySym.setTo( 0.0 );
    cv::gpu::multiply( itsLabResponse[0], sw->l, itsGpuBuffer );
    cv::gpu::add( itsSaliencySym, itsGpuBuffer, itsSaliencySym );

    cv::gpu::multiply( itsLabResponse[1], sw->a, itsGpuBuffer );
    cv::gpu::add( itsSaliencySym, itsGpuBuffer, itsSaliencySym );

    cv::gpu::multiply( itsLabResponse[2], sw->b, itsGpuBuffer );
    cv::gpu::add( itsSaliencySym, itsGpuBuffer, itsSaliencySym );

    cv::gpu::multiply( itsOResponse, sw->o, itsGpuBuffer );
    cv::gpu::add( itsSaliencySym, itsGpuBuffer, itsSaliencySym );

    cv::gpu::multiply( itsSResponse, sw->s, itsGpuBuffer );
    cv::gpu::add( itsSaliencySym, itsGpuBuffer, itsSaliencySym );

		cv::Mat temp( itsSaliencySym.size(), itsSaliencySym.type() );
		temp = itsSaliencySym;
		itsLabFFT[0] = padImageFFT( temp, itsMaxFilterSize, itsFFTSizeSpatial );

    itsNormalizeIterSpatial( itsLabFFT[0], itsSaliencySym, itsSplitBufferSpatial, 0 );

    //maxNormalize( itsSaliencySym );
  }

  // Download everything
  if( lw || sw )
  {
    stream.enqueueDownload( itsLabResponse[0], res.lResponse );
    stream.enqueueDownload( itsLabResponse[1], res.aResponse );
    stream.enqueueDownload( itsLabResponse[2], res.bResponse );
  }

  if( lw )
    stream.enqueueDownload( itsSaliencyLab, res.labSaliency );

  if( rw )
  {
    stream.enqueueDownload( itsBgrResponse[0], res.iResponse );
    stream.enqueueDownload( itsBgrResponse[1], res.rgResponse );
    stream.enqueueDownload( itsBgrResponse[2], res.byResponse );

    stream.enqueueDownload( itsSaliencyBgr, res.rgbSaliency );
  }

  if( sw )
  {
    stream.enqueueDownload( itsSResponse, res.sResponse );

    stream.enqueueDownload( itsSaliencySym, res.symSaliency );
  }

  stream.enqueueDownload( itsOResponse, res.oResponse );

  stream.waitForCompletion();

  return res;
}

void sg::SalientGreenGPU::allocateGPUBuffers( bool rw, bool lw, bool sw )
{
  std::cout << "Allocating memory on the GPU.\n";

  // We force a release of all prior stuff just in case OpenCV isn't
  // deallocating things when we want more space on the GPU

  if( rw || sw )
  {
    // LAB response is same size as input, one channel floating point
		for( auto & i : itsLabResponse )
      i = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  }
  if( rw )
  {
    // RGB response is same size as input, one channel floating point
		for( auto & i : itsBgrResponse )
      i = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  }

  // O and S response is same size as input, one channel floating point
  itsOResponse = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  itsSResponse = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );

  if( lw || sw )
  {
    // FFT lab are sized according to fftsizespatial, two channels
		for( auto & i : itsLabFFT )
      i = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC2 );
  }
  if( rw )
  {
    // RGB lab are sized according to fftsizespatial, two channels
		for( auto & i : itsBgrFFT )
      i = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC2 );
  }

  // FFT for grayscale is sized according to fftsizeFrequency, two channels
  itsGrayscaleFFT = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );
  itsOBuffer = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );
  itsSBuffer = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC1 );

  // Buffer used for splitting images
  itsSplitBufferFrequency.resize( 2 );
	for( auto & i : itsSplitBufferFrequency )
    i = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC1 );

  itsSplitBufferSpatial.resize( 2 );
	for( auto & i : itsSplitBufferSpatial )
    i = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC1 );

  itsSaliencyBgr = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  itsSaliencyLab = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  itsSaliencySym = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
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

void sg::SalientGreenGPU::doSymmetryAndGaborGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & gabor )
{
  itsLogGabor.getSymmetryAndEdgeResponses( fftImage, itsGpuBufferFrequency[0], itsSplitBufferFrequency, itsDoGBankFrequency );

  cv::gpu::dft( itsGpuBufferFrequency[0], gabor, gabor.size(), cv::DFT_INVERSE );
  absFC2Wrapper( gabor );
}

void sg::SalientGreenGPU::doDoGGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & output )
{
  itsDoGBank( fftImage, itsGpuBufferSpatial );

  cv::gpu::dft( itsGpuBufferSpatial, output, output.size(), cv::DFT_INVERSE );
  absFC2Wrapper( output );
}
