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

#ifndef SALIENTGREEN_SALIENTGREENGPU_H_
#define SALIENTGREEN_SALIENTGREENGPU_H_

#include <SalientGreenGPU/Filters/LogGabor.H>
#include <SalientGreenGPU/Filters/Gaussian.H>
#include <SalientGreenGPU/Normalization/Normalization.H>
#include <string>
#include <array>

namespace sg
{
  class SalientGreenGPU
  {
    public:
      //! Contains the results from computing saliency
      struct Results
      {

        cv::Mat lResponse;
        cv::Mat aResponse;
        cv::Mat bResponse;

        cv::Mat oResponse;

        cv::Mat labSaliency;
        
				Results() {}

				Results( cv::Size sz ) : 
					lResponse( sz, CV_32FC1 ),
					aResponse( sz, CV_32FC1 ),
					bResponse( sz, CV_32FC1 ),
					oResponse( sz, CV_32FC1 ),
					labSaliency( sz, CV_32FC1 )
				{}

        //! Displays all results on screen
        void show();
        //! Saves all results to images
        void save( std::string const & prefix, std::string const & filetype );
      };

      //! Weights for computing LAB saliency
      struct labWeights
      {
        labWeights() : l( 1 ), a( 0.5 ), b( 0.5 ), o( 1 )
        { }

        labWeights( double l_, double a_, double b_, double o_ ) : l( l_ ), a( a_ ), b( b_ ),
          o( o_ )
        { }

        double l;
        double a;
        double b;
        double o;
      };

      //! Initializes Salient Green and precomputes several filters
      SalientGreenGPU();

			//! Releases all GPU data
			void release();

      //! Computes a saliency map for a new image using an FFT based approach on the GPU
      /*! @param[in] image An RGB CV_8U3 formatted image
          @param[in] lw The lab weights.  Pass nullptr to disable
          @return A saliency map */
      Results computeSaliencyGPU( cv::Mat const & image,
                                  labWeights const * lw = nullptr );

    protected:
      //! Prepares all filters needed by saliency
      void prepareFiltersGPU();

			//! Allocates all buffers that we'll need
			void allocateGPUBuffers();

      //! Calculates an orientation conspicuity map
			/*! @param[in] fftImage SIngle channel grayscale frequency domain image
			 		@param[out] output A single channel pre-allocated matrix */
      void doGaborGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & output );

      //! Calculates a map using a center surround bank
			/*! @param[in] fftImage A single channel frequency domain image
			 		@param[out] output A single channel pre-allocated matrix */
      void doDoGGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & output );

    private:
      cv::Mat itsInputImage; //!< The input image
			cv::gpu::GpuMat itsInputImageGPU; //!< The input image on the GPU

			//======= FFT Stuff =============
      cv::Size itsFFTSizeSpatial;
      cv::Size itsFFTSizeFrequency;
      cv::Size itsMaxFilterSize;
      LogGabor itsLogGabor; //!< Log Gabor filter bank
      DoGCenterSurround itsDoGBank; //!< Center surround operation for color channels
      DoGCenterSurround itsDoGBankFrequency; //!< Handles DoG operation on gabor results
      NormalizeIterative itsNormalizeIterSpatial;
      NormalizeIterative itsNormalizeIterFrequency;

			// GPU Buffers
			cv::gpu::GpuMat itsSaliencyLab;
			cv::gpu::GpuMat itsGpuBuffer; //!< Sized same as image, one channel
			cv::gpu::GpuMat itsGpuBufferSpatial; //!< Used in spatial calculations, two channels
			std::array<cv::gpu::GpuMat, 2> itsGpuBufferFrequency; //!< Used in frequency calculations, two channels each
		 	std::vector<cv::gpu::GpuMat> itsSplitBufferSpatial; //!< Pre-allocated buffer for splitting data
		 	std::vector<cv::gpu::GpuMat> itsSplitBufferFrequency; //!< Pre-allocated buffer for splitting data
			std::array<cv::gpu::GpuMat, 3> itsLabResponse;
			std::array<cv::gpu::GpuMat, 3> itsLabFFT;
			cv::gpu::GpuMat itsOResponse;
			cv::gpu::GpuMat itsSResponse;
			cv::gpu::GpuMat itsGrayscaleFFT;
			cv::gpu::GpuMat itsOBuffer;
  };
}

#endif // SALIENTGREENGPU_SALIENTGREEN_H_
