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

#ifndef SALIENTGREENGPU_FILTERS_GAUSSIAN_H_
#define SALIENTGREENGPU_FILTERS_GAUSSIAN_H_

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace sg
{
	//! Create a 1D separable gaussian kernel in the spatial domain
	/*! @param[in] peak maximum value
	 		@param[in] sigma standard deviation of filter
			@param[in] maxhw maximum height or width of the filter
			@param[in] threshPercent used to determine half width of the filter
			@return the 1D gaussian filter */
	cv::Mat gaussian( double peak, double sigma, int maxhw, double threshPercent = 1.0 );

	//! Create a 2D isotropic gaussian kernel in the spatial domain
	cv::Mat gaussian2D( double peak, double sigma, int maxhw, double threshPercent = 1.0 );

	//! A class for representing a filter bank of difference of gaussian filters in the frequency domain
	class DoGCenterSurround
	{
		public:
			//! Create an invalid center surround
			DoGCenterSurround();

			//! Create a center surround filter bank
			/*! @param[in] maxhw Maximum height or width for filter in spatial domain 
			    @param[n] size Total padded size for filters
			 		@param[in] initSigma Sigma for the first gaussian (was 4)
					@param[in] mult multiplicative difference between wide and narrow gaussians that compose DoG
					@param[in] scales Number of DoG filters to generate (was 4) */
			DoGCenterSurround( int maxhw, cv::Size const & size, double initSigma = 1.6, double mult = 1.6, int scales = 3, double scaleFactor = 1.6 );

			void release();

			//! Applies the filter bank to the input image in the frequency domain
			/*! @param[in] input Frequency domain representation of image
			 		@param[out] output Frequency domain representation of result */
			void operator()( cv::gpu::GpuMat const & input, cv::gpu::GpuMat & output );

			//! Calculates the largest filter size possible
			static int largestFilterSize( int maxhw, double initSigma = 1.6, double mult = 1.6, int scales = 3, double scaleFactor = 1.6 );

		private:
			bool itsIsValid; //!< Whether we have valid filters
			std::vector<cv::gpu::GpuMat> itsFilters; //!< DoG Filters on the GPU (one for each scale, CV32FC2 format)
			cv::gpu::GpuMat itsBuffer;

			/* total gpu size:
			 *  (scales + 1 ) * size
			 */

	};
}

#endif // SALIENTGREENGPU_FILTERS_LOWPASS_H_
