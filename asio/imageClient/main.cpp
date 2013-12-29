//
// blocking_tcp_echo_client.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using boost::asio::ip::tcp;

enum { max_length = 1024 };

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 3)
    {
      std::cerr << "Usage: blocking_tcp_echo_client <host> <port>\n";
      return 1;
    }

    boost::asio::io_service io_service;

    tcp::resolver resolver(io_service);
    tcp::resolver::query query(tcp::v4(), argv[1], argv[2]);
    tcp::resolver::iterator iterator = resolver.resolve(query);

    tcp::socket s(io_service);
    s.connect(*iterator);

    using namespace std; // For strlen.
    char reply[max_length];

	boost::asio::deadline_timer t(io_service);
	
	const int bucketSize = 64;
	const int imageSizeX = 400;
	const int imageSizeY = 300;
	int rect[4];
				
	for(int by = 0; by <= imageSizeY/bucketSize; by++) {
		rect[2] = by * bucketSize;
		rect[3] = rect[2] + bucketSize - 1;
		if(rect[3] > imageSizeY - 1) rect[3] = imageSizeY - 1;
		for(int bx = 0; bx <= imageSizeX/bucketSize; bx++) {
			rect[0] = bx * bucketSize;
			rect[1] = rect[0] + bucketSize - 1;
			if(rect[1] > imageSizeX - 1) rect[1] = imageSizeX - 1;
			
			const float grey = (float)(random() % 457) / 457.f;
			std::cout<<"grey"<<grey<<"\n";

			const unsigned npix = (rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1);
			std::cout<<"n pixels "<<npix<<"\n";
			int npackage = npix * 16 / 4096;
			if((npix * 16) % 4096 > 0) npackage++;
			std::cout<<"n packages "<<npackage<<"\n";
			
			boost::asio::write(s, boost::asio::buffer((char *)rect, 16));
			boost::array<char, 128> buf;
			boost::system::error_code error;
			size_t reply_length = s.read_some(boost::asio::buffer(buf), error);

			std::cout<<" bucket("<<rect[0]<<","<<rect[1]<<","<<rect[2]<<","<<rect[3]<<")\n";
			
			std::cout<<"extended pixels"<<npackage * 256<<"\n";
			
			float *color = new float[npackage * 256 * 4];
			for(int i = 0; i < npix; i++) {
				color[i * 4] = color[i * 4 + 1] = color[i * 4 + 2] = grey;
				color[i * 4 + 3] = 1.f;
			}
			
			for(int i=0; i < npackage; i++) {
				boost::asio::write(s, boost::asio::buffer((char *)&(color[i * 256 * 4]), 4096));
				reply_length = s.read_some(boost::asio::buffer(buf), error);
			}
			
			boost::asio::write(s, boost::asio::buffer("transferEnd", 11));
			reply_length = s.read_some(boost::asio::buffer(buf), error);
			
			t.expires_from_now(boost::posix_time::seconds(1));
			t.wait();
			
		}
	}
	
	s.close();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}