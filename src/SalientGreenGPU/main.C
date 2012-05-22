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

// #define BOOST_FILESYSTEM_VERSION 2
// #include <boost/filesystem.hpp>

int main( int argc, char ** argv )
{
  sg::SalientGreenGPU green;
  // Timer t;

  // Run on every file in directory, output in another directory
  sg::SalientGreenGPU::labWeights lw;

  //sw.s = 2;

  if( argc > 2 )
  {
    if( std::string( argv[1] ) == "file" )
    {
      cv::Mat input;
      sg::SalientGreenGPU::Results results;

      input = cv::imread( argv[2] );
      results = green.computeSaliencyGPU( input,
                                          &lw );

			// Timing version, let it run once to warm up the GPU
			/*t.start();
      results = green.computeSaliencyGPU( input,
                                          &rw,
                                          &lw,
                                          &sw );
			t.stop();*/

      results.save( "~/Code/SalientGreenGPU/test/", "jpg" );
      cv::imwrite( "~/Code/SalientGreenGPU/test/input.jpg", input );
    }
    else if( std::string( argv[1] ) == "directory" )
    {
      // int index = 0;
      // using namespace boost::filesystem2;
      // std::string ext( argv[2] );
      // path input( argv[3] );
      // path outputRW( argv[4] );
      // path outputLW( argv[5] );
      // path outputSW( argv[6] );
      // cv::Mat inputIm;

      // if( exists( input ) && exists( outputRW ) && exists( outputLW ) && exists( outputSW ) )
      // {
      //   directory_iterator end;

      //   for( directory_iterator iter( input ); iter != end; ++iter, ++index )
      //   {
      //     if( index == 1 )
      //       t.start();

      //     std::string filePrefix = iter->path().parent_path().string() + "/";
      //     std::string file = iter->path().stem();
      //     std::string extension = iter->path().extension();

      //     if( ext != extension )
      //       continue;
      //     if( exists( path( outputRW.string() + iter->path().stem() + iter->path().extension() ) ) )
      //     {
      //       std::cout << "Skipping " << file << std::endl;
      //       continue;
      //     }

      //     std::cout << "Analyzing " << file << std::endl;
      //     inputIm = cv::imread( iter->path().string() );
      //     bool resize = false;

      //     int size = 9999;

      //     if( inputIm.rows > size || inputIm.cols > size )
      //     {
      //       std::cout << "Input image was too big, resizing to 50%" << std::endl;
      //       resize = true;
      //       sg::resizeInPlace( inputIm, cv::Size(), 0.5, 0.5 );
      //     }

      //     sg::SalientGreenGPU::Results results;

      //     try
      //     {
      //       results = green.computeSaliencyGPU( inputIm, &rw, &lw, &sw );
      //     }
      //     catch( ... )
      //     {
      //       std::cout << "Some kind of runtime error, trying again" << std::endl;
      //       try
      //       {
      //         green.release();
      //         green = sg::SalientGreenGPU();
      //         results = green.computeSaliencyGPU( inputIm, &rw, &lw, &sw );
      //       }
      //       catch( ... )
      //       {
      //         std::cout << "Couldn't recover, skipping image" << std::endl;
      //         continue;
      //       }
      //     }

      //     if( resize )
      //     {
      //       sg::resizeInPlace( results.rgbSaliency, cv::Size(), 2, 2 );
      //       sg::resizeInPlace( results.labSaliency, cv::Size(), 2, 2 );
      //       sg::resizeInPlace( results.symSaliency, cv::Size(), 2, 2 );
      //     }

      //     cv::imwrite( outputRW.string() + file + extension, results.rgbSaliency );
      //     cv::imwrite( outputLW.string() + file + extension, results.labSaliency );
      //     cv::imwrite( outputSW.string() + file + extension, results.symSaliency );
      //   }
      // }
      // else
      //   std::cerr << "Directories do not exist\n";

      // t.stop();
    } // end if directory arg
  } // if args

  return 0;
}
