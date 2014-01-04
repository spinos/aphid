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
#include <boost/thread/mutex.hpp>

boost::mutex io_mutex;

using boost::asio::ip::tcp;
BaseServer::BaseServer(short port) 
{
	m_port = port;
}

BaseServer::~BaseServer() {}

void BaseServer::start()
{
	std::cout<<" start server at port" << m_port<<" ";
	boost::thread t(boost::bind(&BaseServer::server, this, m_port));
}

void BaseServer::server(short port)
{
	
	boost::asio::io_service io_service;
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
	io_mutex.lock();
	std::clog<<"connection opened\n";
	
	for (;;) {
	    
		boost::array<char, 4096> buf;
      boost::system::error_code error;
      size_t length = sock->read_some(boost::asio::buffer(buf), error);
      if (error == boost::asio::error::eof) {
		// Connection closed cleanly by peer.
		std::clog<<"connection closed\n";
		break;
	  }

      else if (error)
        throw boost::system::system_error(error); // Some other error.

		//std::cout<<"thread id "<<boost::this_thread::get_id()<<"\n";
		processRead(buf.data(), length);
		
		boost::asio::write(*sock, boost::asio::buffer("beep.", 5));
		
    }
    io_mutex.unlock();
  }
  catch (std::exception& e) {
    std::cerr << "Exception in session: " << e.what() << "\n";
    io_mutex.unlock();
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