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
#include <SalientGreenGPU/Filters/Utility.H>
#include <SalientGreenGPU/Timer.H>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cvblob.h>

// #define BOOST_FILESYSTEM_VERSION 2
// #include <boost/filesystem.hpp>

sg::SalientGreenGPU::Results runSaliencyAndBlobs(cv::Mat img)
{
  sg::SalientGreenGPU::Results results;
   results = green.computeSaliencyGPU( input,
                  &lw );

  /////// do connected components
  cvb::CvBlobs blobs;
  unsigned int result = cvb::cvLabel(grey, labelImg, blobs);
  for (cvb::CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it)
  {
    std::printf("found blob: xy_orig(%d,%d) xy_max(%d,%d)",(*it).minx,(*it).miny,(*it).maxx,(*it).maxy);
  }
  ////// end connected components

  return results;
}

int main( int argc, char ** argv )
{
  std::cout << "SalientGreenGPU launched\n";
  sg::SalientGreenGPU green;
  // Timer t;

  // Run on every file in directory, output in another directory
  sg::SalientGreenGPU::labWeights lw;
      
  cv::Mat input;
  sg::SalientGreenGPU::Results results;

  input = cv::Mat(cvLoadImage( "/Users/lewisanderson/Code/Github/SalientGreenGPU/test/test.jpg"));
  // std::cout << input << "\n";
  // cv::imshow("input", input);
  // cvWaitKey(0);

  std::cout << "SalientGreenGPU image loaded depth=" << input.depth() << " (CV_32F=" << CV_32F << ") (CV_8U=" << CV_8U << ") scn=" <<  input.channels() <<"\n"; // LEWIS DEBUG

  results = runSaliencyAndBlobs(input);
 

  std::cout << "SalientGreenGPU done computing! saving ...\n";

  // Timing version, let it run once to warm up the GPU
  /*t.start();
  results = green.computeSaliencyGPU( input,
                  &rw,
                  &lw,
                  &sw );
  t.stop();*/

  results.save( "~/Code/SalientGreenGPU/test/", "jpg" );
  cv::imwrite( "~/Code/SalientGreenGPU/test/input.jpg", input );

  if( argc > 2 )
  {
    if( std::string( argv[1] ) == "file" )
    {
         //    cv::Mat input;
         //    sg::SalientGreenGPU::Results results;

         //    input = cv::imread( argv[2] );
         //    results = green.computeSaliencyGPU( input,
         //                                        &lw );

      			// // Timing version, let it run once to warm up the GPU
      			// /*t.start();
         //    results = green.computeSaliencyGPU( input,
         //                                        &rw,
         //                                        &lw,
         //                                        &sw );
      			// t.stop();*/

         //    results.save( "~/Code/SalientGreenGPU/test/", "jpg" );
         //    cv::imwrite( "~/Code/SalientGreenGPU/test/input.jpg", input );
    }
    else if( std::string( argv[1] ) == "directory" )
    {
    } // end if directory arg
  } // if args

  return 0;
}
