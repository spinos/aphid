/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 3/13/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */


#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include <boost/asio.hpp>
#include <xmlrpc.h>
#include <xmlrpc_client.h>
#include <xmlrpc-c/girerr.hpp>
#include <xmlrpc-c/base.hpp>
#include <xmlrpc-c/client.hpp>
#include <xmlrpc-c/client_transport.hpp>

using boost::asio::ip::tcp;

int testAsio(const char * hostURL, const char * dir)
{
	try {
	boost::asio::io_service io_service;

    // Get a list of endpoints corresponding to the server name.
    tcp::resolver resolver(io_service);
    tcp::resolver::query query(hostURL, "http");
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Try each endpoint until we successfully establish a connection.
    tcp::socket socket(io_service);
    boost::asio::connect(socket, endpoint_iterator);

    // Form the request. We specify the "Connection: close" header so that the
    // server will close the socket after transmitting the response. This will
    // allow us to treat all data up until the EOF as the content.
    boost::asio::streambuf request;
    std::ostream request_stream(&request);
    request_stream << "POST " << dir << " HTTP/1.1\r\n";
	request_stream << "User-Agent: Mozilla/4.0 (compatible; MSIE5.01; Windows NT)\r\n";
    request_stream << "Host: " << hostURL << "\r\n";
    request_stream << "Accept-Language: en-us\r\n";
	request_stream << "Accept-Encoding: gzip, deflate\r\n";
    request_stream << "Connection: Keep-Alive\r\n\r\n";
	
    // Send the request.
    boost::asio::write(socket, request);

    // Read the response status line. The response streambuf will automatically
    // grow to accommodate the entire line. The growth may be limited by passing
    // a maximum size to the streambuf constructor.
    boost::asio::streambuf response;
    boost::asio::read_until(socket, response, "\r\n");

    // Check that response is OK.
    std::istream response_stream(&response);
    std::string http_version;
    response_stream >> http_version;
    unsigned int status_code;
    response_stream >> status_code;
    std::string status_message;
    std::getline(response_stream, status_message);
    if (!response_stream || http_version.substr(0, 5) != "HTTP/")
    {
      std::cout << "Invalid response\n";
      return 1;
    }
    if (status_code != 200)
    {
      std::cout << "Response returned with status code " << status_code << "\n";
      return 1;
    }

    // Read the response headers, which are terminated by a blank line.
    boost::asio::read_until(socket, response, "\r\n\r\n");

    // Process the response headers.
    std::string header;
    while (std::getline(response_stream, header) && header != "\r")
      std::cout << header << "\n";
    std::cout << "\n";

    // Write whatever content we already have to output.
    if (response.size() > 0)
      std::cout << &response;

    // Read until EOF, writing data to output as we go.
    boost::system::error_code error;
    while (boost::asio::read(socket, response,
          boost::asio::transfer_at_least(1), error))
      std::cout << &response;
    if (error != boost::asio::error::eof)
      throw boost::system::system_error(error);
  }
  catch (std::exception& e)
  {
    std::cout << "Exception: " << e.what() << "\n";
  }
  
	return 0;
}

static void
die_if_fault_occurred(xmlrpc_env * const envP) {
    if (envP->fault_occurred) {
        fprintf(stderr, "XML-RPC Fault: %s (%d)\n",
                envP->fault_string, envP->fault_code);
        exit(1);
    }
}

int testRPC()
{
	xmlrpc_c::clientXmlTransport_curl transport;
	xmlrpc_c::client_xml client = xmlrpc_c::client_xml(&transport);
	xmlrpc_c::carriageParm_curl0 myCarriageParm("shotgun.of3d.com");
	return 0;
}

int main(int argc, char* argv[])
{
  
    if (argc != 3)
    {
      std::cout << "Usage: sync_client <server> <path>\n";
      std::cout << "Example:\n";
      std::cout << "  sync_client www.boost.org /LICENSE_1_0.txt\n";
      return 1;
    }

	//testAsio(argv[1], argv[2]);
	testRPC();
    
  return 0;
}