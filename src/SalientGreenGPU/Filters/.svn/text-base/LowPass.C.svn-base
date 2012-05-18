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

#include <SalientGreenGPU/Filters/LowPass.H>
#include <SalientGreenGPU/Filters/Utility.H>

cv::Mat sg::lowpass( int width, int height, double cutoff, int order )
{
  cv::Mat filter( height, width, CV_32FC1 );
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

  for( int i = 0; i < height; ++i )
    for( int j = 0; j < width; ++j )
    {
      val = std::sqrt( xData[j] * xData[j] + yData[i] * yData[i] );
      val /= cutoff;
      val = std::pow( val, 2 * order );
      val = 1.0 / ( 1.0 + val );
      filter.at<float>( i, j ) = val;
    }

  ifftshift( filter );

  return filter;
}

namespace
{
	/*! Adapted from the iLab Neuromorphic Robotics Toolkit (NRT)
	 * Copyright 2010 by the University of Southern California (USC) and
	 * the iLab at USC.
	 *
	 * See http://ilab.usc.edu for information about this project
	 * and licensing information */
  cv::Mat lowpass9x( cv::Mat const & input )
  {
    const int w = input.cols;
    const int h = input.rows;

    cv::Mat result = input;
    if (w < 2) return input;

    auto sptr = input.begin<float>();
    auto dptr = result.begin<float>();

    // boundary conditions: truncated filter
    for (int j = 0; j < h; j ++)
    {
      // leftmost points
      *dptr++ = sptr[0] * (70.0F / 163.0F) +
                sptr[1] * (56.0F / 163.0F) +
                sptr[2] * (28.0F / 163.0F) +
                sptr[3] * ( 8.0F / 163.0F) +
                sptr[4] * ( 1.0F / 163.0F);
      *dptr++ = (sptr[0] + sptr[2]) * (56.0F / 219.0F) +
                sptr[1] * (70.0F / 219.0F) +
                sptr[3] * (28.0F / 219.0F) +
                sptr[4] * ( 8.0F / 219.0F) +
                sptr[5] * ( 1.0F / 219.0F);
      *dptr++ = (sptr[0] + sptr[4]) * (28.0F / 247.0F) +
                (sptr[1] + sptr[3]) * (56.0F / 247.0F) +
                sptr[2] * (70.0F / 247.0F) +
                sptr[5] * ( 8.0F / 247.0F) +
                sptr[6] * ( 1.0F / 247.0F);
      *dptr++ = (sptr[0] + sptr[6]) * ( 8.0F / 255.0F) +
                (sptr[1] + sptr[5]) * (28.0F / 255.0F) +
                (sptr[2] + sptr[4]) * (56.0F / 255.0F) +
                sptr[3] * (70.0F / 255.0F) +
                sptr[7] * ( 1.0F / 255.0F);

      // far from the borders
      for (int i = 0; i < w - 8; i ++)
      {
        *dptr++ = (sptr[0] + sptr[8]) * ( 1.0F / 256.0F) +
                  (sptr[1] + sptr[7]) * ( 8.0F / 256.0F) +
                  (sptr[2] + sptr[6]) * (28.0F / 256.0F) +
                  (sptr[3] + sptr[5]) * (56.0F / 256.0F) +
                  sptr[4] * (70.0F / 256.0F);
        sptr ++;
      }

      // rightmost points
      *dptr++ = sptr[0] * ( 1.0F / 255.0F) +
                (sptr[1] + sptr[7]) * ( 8.0F / 255.0F) +
                (sptr[2] + sptr[6]) * (28.0F / 255.0F) +
                (sptr[3] + sptr[5]) * (56.0F / 255.0F) +
                sptr[4] * (70.0F / 255.0F);
      sptr ++;
      *dptr++ = sptr[0] * ( 1.0F / 247.0F) +
                sptr[1] * ( 8.0F / 247.0F) +
                (sptr[2] + sptr[6]) * (28.0F / 247.0F) +
                (sptr[3] + sptr[5]) * (56.0F / 247.0F) +
                sptr[4] * (70.0F / 247.0F);
      sptr ++;
      *dptr++ = sptr[0] * ( 1.0F / 219.0F) +
                sptr[1] * ( 8.0F / 219.0F) +
                sptr[2] * (28.0F / 219.0F) +
                (sptr[3] + sptr[5]) * (56.0F / 219.0F) +
                sptr[4] * (70.0F / 219.0F);
      sptr ++;
      *dptr++ = sptr[0] * ( 1.0F / 163.0F) +
                sptr[1] * ( 8.0F / 163.0F) +
                sptr[2] * (28.0F / 163.0F) +
                sptr[3] * (56.0F / 163.0F) +
                sptr[4] * (70.0F / 163.0F);
      sptr += 5;  // sptr back to same as dptr (start of next line)
    }

    return result;
  }

	/*! Adapted from the iLab Neuromorphic Robotics Toolkit (NRT)
	 * Copyright 2010 by the University of Southern California (USC) and
	 * the iLab at USC.
	 *
	 * See http://ilab.usc.edu for information about this project
	 * and licensing information */
  cv::Mat lowpass9y( cv::Mat const & input )
  {
    const int w = input.cols;
    const int h = input.rows;

    cv::Mat result = input;
    if (h < 2) return input;

    auto sptr = input.begin<float>();
    auto dptr = result.begin<float>();

    // *** vertical pass ***
    int const w2 = w + w, w3 = w2 + w, w4 = w3 + w, w5 = w4 + w, w6 = w5 + w, w7 = w6 + w,  w8 = w7 + w;
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = sptr[ 0] * (70.0F / 163.0F) +
                sptr[ w] * (56.0F / 163.0F) +
                sptr[w2] * (28.0F / 163.0F) +
                sptr[w3] * ( 8.0F / 163.0F) +
                sptr[w4] * ( 1.0F / 163.0F);
      sptr ++;
    }
    sptr -= w; // back to top-left
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = (sptr[ 0] + sptr[w2]) * (56.0F / 219.0F) +
                sptr[ w] * (70.0F / 219.0F) +
                sptr[w3] * (28.0F / 219.0F) +
                sptr[w4] * ( 8.0F / 219.0F) +
                sptr[w5] * ( 1.0F / 219.0F);
      sptr ++;
    }
    sptr -= w; // back to top-left
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = (sptr[ 0] + sptr[w4]) * (28.0F / 247.0F) +
                (sptr[ w] + sptr[w3]) * (56.0F / 247.0F) +
                sptr[w2] * (70.0F / 247.0F) +
                sptr[w5] * ( 8.0F / 247.0F) +
                sptr[w6] * ( 1.0F / 247.0F);
      sptr ++;
    }
    sptr -= w; // back to top-left
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = (sptr[ 0] + sptr[w6]) * ( 8.0F / 255.0F) +
                (sptr[ w] + sptr[w5]) * (28.0F / 255.0F) +
                (sptr[w2] + sptr[w4]) * (56.0F / 255.0F) +
                sptr[w3] * (70.0F / 255.0F) +
                sptr[w7] * ( 1.0F / 255.0F);
      sptr ++;
    }
    sptr -= w;   // back to top-left
    for (int j = 0; j < h - 8; j ++)
      for (int i = 0; i < w; i ++)
      {
        *dptr++ = (sptr[ 0] + sptr[w8]) * ( 1.0F / 256.0F) +
                  (sptr[ w] + sptr[w7]) * ( 8.0F / 256.0F) +
                  (sptr[w2] + sptr[w6]) * (28.0F / 256.0F) +
                  (sptr[w3] + sptr[w5]) * (56.0F / 256.0F) +
                  sptr[w4]  * (70.0F / 256.0F);
        sptr ++;
      }
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = sptr[ 0] * ( 1.0F / 255.0F) +
                (sptr[ w] + sptr[w7]) * ( 8.0F / 255.0F) +
                (sptr[w2] + sptr[w6]) * (28.0F / 255.0F) +
                (sptr[w3] + sptr[w5]) * (56.0F / 255.0F) +
                sptr[w4] * (70.0F / 255.0F);
      sptr ++;
    }
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = sptr[ 0] * ( 1.0F / 247.0F) +
                sptr[ w] * ( 8.0F / 247.0F) +
                (sptr[w2] + sptr[w6]) * (28.0F / 247.0F) +
                (sptr[w3] + sptr[w5]) * (56.0F / 247.0F) +
                sptr[w4] * (70.0F / 247.0F);
      sptr ++;
    }
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = sptr[ 0] * ( 1.0F / 219.0F) +
                sptr[ w] * ( 8.0F / 219.0F) +
                sptr[w2] * (28.0F / 219.0F) +
                (sptr[w3] + sptr[w5]) * (56.0F / 219.0F) +
                sptr[w4] * (70.0F / 219.0F);
      sptr ++;
    }
    for (int i = 0; i < w; i ++)
    {
      *dptr++ = sptr[ 0] * ( 1.0F / 163.0F) +
                sptr[ w] * ( 8.0F / 163.0F) +
                sptr[w2] * (28.0F / 163.0F) +
                sptr[w3] * (56.0F / 163.0F) +
                sptr[w4] * (70.0F / 163.0F);
      sptr ++;
    }
    return result;
  }
}

cv::Mat sg::lowpass9( cv::Mat const & input )
{
  cv::Mat result = input;
  result = lowpass9x( result );
  result = lowpass9y( result );

  return result;
}

