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
    
	boost::asio::io_service i;
	
	boost::asio::deadline_timer t(i);
	for(int l = 0; l < 10; l++) {
		boost::posix_time::ptime tt(boost::posix_time::second_clock::local_time());

		std::string tts = to_iso_extended_string(tt);
		size_t request_length = tts.length();
		boost::asio::write(s, boost::asio::buffer(tts.c_str(), request_length));

		size_t reply_length = boost::asio::read(s, boost::asio::buffer(reply, 5));
		std::cout << "Reply is: ";
		std::cout.write(reply, reply_length);
		std::cout << "\n";
		
		t.expires_from_now(boost::posix_time::seconds(1));
		t.wait();
		std::clog<<" loop "<<l<<"\n";
	}
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}