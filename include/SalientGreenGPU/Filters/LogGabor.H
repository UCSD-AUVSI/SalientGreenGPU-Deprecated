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

#ifndef SALIENTGREENGPU_FILTERS_LOGGABOR_H_
#define SALIENTGREENGPU_FILTERS_LOGGABOR_H_

#include <SalientGreenGPU/Filters/Gaussian.H>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <cmath> // for M_PI
#include <array>

namespace sg
{
  //! A class for creating a bank of log gabor filters
  class LogGabor
  {
    public:

      //! Buffers used by this class
      /*struct Buffers
      {
        Buffers( cv::gpu::GpuMat & _fc1_1,
                 cv::gpu::GpuMat & _fc1_2,
                 cv::gpu::GpuMat & _fc2_1,
                 cv::gpu::GpuMat & _fc2_2,
                 std::vector<cv::gpu::GpuMat> & _split ) :
          fc1_1( _fc1_1 ), fc1_2( _fc1_2 ),
          fc2_1( _fc2_1 ), fc2_2( _fc2_2 ),
          splitBuffer( _split )
        { }

        cv::gpu::GpuMat & summedMag;
        cv::gpu::GpuMat & summedEnergy;
        cv::gpu::GpuMat & dftBuffer1;
        cv::gpu::GpuMat & dftBuffer2;
        std::vector<cv::gpu::GpuMat> & splitBuffer;
      }; */

      //! Create an empty gabor object - can't be used for anything
      LogGabor();

			//! Releases all GPU data
			/*! Note that this is only here until memory issues are taken care of */
			void release();

      //! Prepares for log gabor filter creation
      /*! Does some initial calculations involved in creating Log Gabor filters
          for an image of a given size
          @param[in] width filter width
          @param[in] height filter height
          @param[in] lowpass Gaussian envelope applied to all scales
          @param[in] numScales Number of scales to use (was 2)
          @parma[in] minWaveLength Wavelength for smallest scale (was 3)
          @param[in] mult Scaling factor between scales (was 3.00)
          @param[in] sigmaOnf Ratio of standard deviation of Gaussian describing the log gabor filter's transfer function (was 0.55) */
      LogGabor( int width, int height, cv::Mat const & lowpass, int numScales = 2, double minWaveLength = 2.0,
                double mult = 2.00, double sigmaOnf = 0.55 );
      // LEWIS: was numScales=6

			// 7 1.75 .55 20.0 

			/*
			 * Default settings: 6s, mw 3, mult 1.5, sigma .55, std 2.0
			 * mult of 3, stdnoise 2 gives pretty nice settings for symmetrical objects as well */

      //! Adds filters for orientations, assumes that detailed constructor has already been used
      /*! A higher number of orientations will increase our ability to pick up radial symmetry
          @note Allocates the following on the GPU:
            Our filters (numOrientations * numScales)
            Cache for FFT responses
            Cache for IDFT results */
      void addFilters( int numOrientations = 6 );

			//! Gets the edge responses for an input
			/*! @param[in] fftImage The fft image to work on
			 		@param[out] edges The result of running the computation (two channel complex frequency domain result)
					@param[in] splitBuffer A buffer used for splitting channels
					@param[in] dog Center surround processing */
			void getEdgeResponses( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & edges,
					std::vector<cv::gpu::GpuMat> & splitBuffer, DoGCenterSurround & dog );

    protected:
      bool itsValid;
      int itsWidth;
      int itsHeight;
      double itsMult;
      cv::Mat itsSine;
      cv::Mat itsCosine;

      std::vector<cv::Mat> itsGaborScales; //!< Gabor filters before angular component added

      std::vector<std::vector<cv::gpu::GpuMat>> itsFilters; //!< Filters allocated on the GPU

      cv::gpu::GpuMat itsBufferReal; //!< 32FC2
      cv::gpu::GpuMat itsBufferImag; //!< 32FC2
      std::array<cv::gpu::GpuMat, 3> itsFFTBuffer; //!< 32FC2

      /*
       * Total GPU space needed:
       *  4 * fftsize + 10 * fftsize for buffers
       *  numOrient * numScale * 2 * fftsize for filters
       *  = (12 + 2*o*s) * fftsize
       */
  };
}

#endif // SALIENTGREENGPU_FILTERS_LOGGABOR_H_
