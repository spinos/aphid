/*
 *  BaseServer.cpp
 *  aphid
 *
 *  Created by jian zhang on 12/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "BaseServer.h"
#include <boost/thread.hpp>
using boost::asio::ip::tcp;
const int max_length = 262160;
BaseServer::BaseServer(short port) 
{
	boost::asio::io_service io_service;
	server(io_service, port);
}

BaseServer::~BaseServer() {}

void BaseServer::server(boost::asio::io_service& io_service, short port)
{
  tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
  for (;;)
  {
    socket_ptr sock(new tcp::socket(io_service));
    a.accept(*sock);
    boost::thread t(boost::bind(&BaseServer::session, this, sock));
  }
}

void BaseServer::session(socket_ptr sock)
{
	try {
    for (;;) {
      char data[max_length];

      boost::system::error_code error;
      size_t length = sock->read_some(boost::asio::buffer(data), error);
      if (error == boost::asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw boost::system::system_error(error); // Some other error.

		processRead(data, length);
		
		boost::asio::write(*sock, boost::asio::buffer("beep.", 5));
    }
  }
  catch (std::exception& e) {
    std::cerr << "Exception in thread: " << e.what() << "\n";
  }
}

void BaseServer::processRead(const char * data, size_t length)
{
	std::clog<<"received info: "<<data<<" length is "<<length<<"\n";
	boost::asio::io_service i;
	boost::asio::deadline_timer t(i);
	t.expires_from_now(boost::posix_time::seconds(1));
	t.wait();
}