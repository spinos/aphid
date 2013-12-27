/*
 *  BaseClient.h
 *  aphid
 *
 *  Created by jian zhang on 12/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <boost/asio.hpp>
typedef boost::shared_ptr<boost::asio::ip::tcp::tcp::socket> socket_ptr;

class BaseClient {
public:
	BaseClient();
	virtual ~BaseClient();
	
	void connect(const char * host, const char * port);
	
	virtual void contact(const char * data, size_t length);
private:
	socket_ptr m_conn;
};