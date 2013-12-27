/*
 *  BaseClient.cpp
 *  aphid
 *
 *  Created by jian zhang on 12/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <boost/asio.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using boost::asio::ip::tcp;
#include "BaseClient.h"

BaseClient::BaseClient() {}
BaseClient::~BaseClient() {}

void BaseClient::connect(const char * host, const char * port) 
{
	boost::asio::io_service io_service;

    tcp::resolver resolver(io_service);
    tcp::resolver::query query(tcp::v4(), host, port);
    tcp::resolver::iterator iterator = resolver.resolve(query);

    m_conn = socket_ptr(new tcp::socket(io_service));
    m_conn->connect(*iterator);
}

void BaseClient::contact(const char * data, size_t length)
{
	boost::asio::write(*m_conn, boost::asio::buffer(data, length));
	char reply[64];
	size_t reply_length = boost::asio::read(*m_conn, boost::asio::buffer(reply, 5));
	std::cout << "Reply is: ";
	std::cout.write(reply, reply_length);
	std::cout << "\n";
}