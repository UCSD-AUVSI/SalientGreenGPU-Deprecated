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

#ifndef SALIENTGREEN_FILTERS_UTILITY_H_
#define SALIENTGREEN_FILTERS_UTILITY_H_

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace sg
{
  //! Swap the quadrants in a matrix in place
  /*! Has the effect of centering something at (0, 0)
      @param[in] src The input matrix
      @param[out] dest The output matrix. */
  void fftshift( cv::Mat & src );

  //! Undoes an fftshift
  /*! @param[in] src The input matrix
      @param[out] dest The output matrix. */
  void ifftshift( cv::Mat & src );

  //! Calculates the median of a matrix
  float median( cv::Mat const & matrix );

  //! Normalizes the GPU matrix in place
  void normalizeGPU( cv::gpu::GpuMat & mat, float max );

  //! Prepares an image for FFT
  /*! Pads an image by duplicating borders and adding zero valued complex data.  Should be used
      for FFT based convolution where the filters are created in the frequency domain
      @param[in] input The input image
      @param[in] paddedSize The size of the padded image
      @return The padded image, complex valued */
  cv::Mat padImageFFT( cv::Mat const & input, cv::Size const & paddedSize );

  //! Prepares an image for FFT
  /*! Pads an image by duplicating borders in such a way to support cyclic convolution
      Adds zero valued complex data.  Should be used with filters initially created in spatial domain.
      @param[in] input The input image
      @param[in] kernelSize size of largest unpadded kernel
      @param[in] paddedSize The size of the padded image
      @return The padded image, complex valued */
  cv::Mat padImageFFT( cv::Mat const & input, cv::Size const & kernelSize, cv::Size const & paddedSize );

  //! Prepare a kernel for FFT
  /*! The kernel is adjusted such that its center is at (0,0) and filled with zeros appropriately
      @param[in] input the unpadded kernel, real or complex valued
      @param[in] paddedSize the size of the padded kernel
      @return The padded kernel, complex valued */
  cv::Mat padKernelFFT( cv::Mat const & input, cv::Size const & paddedSize );

  //! Gets the padded size for an FFT convolution
  cv::Size getPaddedSize( cv::Size const & imageSize );

  //! Gets the padded size for an FFT convolution
  cv::Size getPaddedSize( cv::Size const & imageSize, cv::Size const & largestKernelSize );

  //! Uncrops the result of an FFT, removing the complex channel and normalizing by
  //! the fft size
  /*! @param[in] input The input FC1 or FC2 matrix
      @param[out] output The output FC1 matrix
      @param[in] splitBuffer buffer with two entries to store results of split operation */
  void uncropFFTGPU( cv::gpu::GpuMat const & input, cv::gpu::GpuMat & output, std::vector<cv::gpu::GpuMat> & splitBuffer );

  void resizeInPlace( cv::Mat & input, cv::Size const & sz = cv::Size(), double dx = 0, double dy = 0 );
}

#endif // SALIENTGREEN_FILTERS_UTILITY_H_
