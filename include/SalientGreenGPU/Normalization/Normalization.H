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

#ifndef SALIENTGREEN_NORMALIZATION_NORMALIZATION_H_
#define SALIENTGREEN_NORMALIZATION_NORMALIZATION_H_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace sg
{
  //! Normalizes using maxnorm as in Itti et al, PAMI 1998 in the spatial domain
	/*! @param[in] input The input image
			@return the normalized image */
  cv::Mat maxNormalize( cv::Mat const & input );

	//! Normalizes using maxnorm on the GPU, in place
  void maxNormalize( cv::gpu::GpuMat & input );

  //! A class for performing iterative normalization in the spatial domain
  /*! Normalizes images using an iterative application
      of a difference of gaussian filter */
  class NormalizeIterative
  {
    public:
      //! Default constructor - does no initialization
      NormalizeIterative();
      //! Create the filters
			/*! @param[in] height height of the images to be filtered
			 		@param[in] width width of the images to be filtered */
			NormalizeIterative( cv::Size const & imageSize, cv::Size const & fftSize );

			void release();

      //! Normalize an image
      /*! @param[in] input the input image
          @param[in] numIter number of times to iterate
          @return the normalized image */
      void operator()( cv::gpu::GpuMat const & input, cv::gpu::GpuMat & output, std::vector<cv::gpu::GpuMat> & splitBuffer, int numIter = 3 );
      
			// Convolution parameters
      static constexpr double inhSig = 40; // 25
      static constexpr double exhSig = 4; // 2
      static constexpr double inhCoef = 1.5; // 1.5
      static constexpr double exhCoef = 0.5; // 0.5
      static constexpr double inhibCoef = 5.0; // 2.0

    private:
      bool itsInitialized;

			cv::Size itsImageSize;
			cv::Size itsFFTSize;

			cv::gpu::GpuMat itsInhibition; //!< Inhibition filter, 32FC2 format, FFTsized
			cv::gpu::GpuMat itsExcitation; //!< Excitation filter, 32FC2 format, FFTsized
			cv::gpu::GpuMat itsResultBuffer; //!< Result buffer, 32FC format, FFTsized
			cv::gpu::GpuMat itsDFTBuffer; //!< DFT result buffer, 32FC2 format, FFTsized
  };
  
	void attenuateBorders( cv::Mat & mat, int size );
}

#endif // SALIENTGREEN_NORMALIZATION_NORMALIZATION_H_
