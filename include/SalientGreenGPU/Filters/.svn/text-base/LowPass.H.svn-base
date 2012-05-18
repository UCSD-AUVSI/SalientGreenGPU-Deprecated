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

#ifndef SALIENTGREENGPU_FILTERS_LOWPASS_H_
#define SALIENTGREENGPU_FILTERS_LOWPASS_H_

#include <opencv2/core/core.hpp>

namespace sg
{
	//! Create a low pass filter in the frequency domain
	/*! @param[in] width width of filter
	 		@param[in] height height of filter
			@param[in] cutoff the cutoff frequency of the filter, 0 to 0.5
			@param[in] order the order of the filter - higher values make the transition sharper;
			order >= 1
			@return The filter */
	cv::Mat lowpass( int width, int height, double cutoff, int order );

	//! Applies a low pass filter of size 9x9 in the spatial domain
	cv::Mat lowpass9( cv::Mat const & input );
}

#endif // SALIENTGREEN_FILTERS_LOWPASS_H_
