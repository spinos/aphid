/*
 *  BaseServer.h
 *  aphid
 *
 *  Created by jian zhang on 12/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>

typedef boost::shared_ptr<boost::asio::ip::tcp::tcp::socket> socket_ptr;

class BaseServer {
public:
	BaseServer(short port);
	virtual ~BaseServer();
	
	void start();
	
protected:
	virtual void processRead(const char * data, size_t length);
private:
	void server(short port);
	void session(socket_ptr sock);
	short m_port;
};