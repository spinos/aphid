/*
 *  MlEngine.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/31/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlEngine.h"
#include "BarbWorks.h"
#include <boost/asio.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time;
using boost::asio::ip::tcp;

MlEngine::MlEngine() 
{
	std::cout<<" renderEngine ";
	m_barb = 0;
}

MlEngine::MlEngine(BarbWorks * w)
{
	std::cout<<" renderEngine ";
	setWorks(w);
}

MlEngine::~MlEngine() 
{
	interruptRender();
}

void MlEngine::setWorks(BarbWorks * w)
{
	m_barb = w;
}

void MlEngine::render() 
{
	interruptRender();
	processBarbs();
	
	m_workingThread = boost::thread(boost::bind(&MlEngine::testOutput, this));
}

void MlEngine::interruptRender()
{
	std::cout<<"treadId"<<m_workingThread.get_id();
	std::cout<<"render cancelled";
	m_workingThread.interrupt();
}

void MlEngine::processBarbs()
{
	boost::timer met;
	met.restart();

	m_barb->createBarbBuffer();
	std::cout<<" barb processed in "<<met.elapsed()<<" seconds\n";
}

void MlEngine::testOutput()
{
	if(!m_barb) return;
	
	ptime tt(second_clock::local_time());
	std::cout<<"test output begins at "<<to_simple_string(tt)<<"\n";
	
    std::string ts("2002-01-20 23:59:59.000");
    ptime tref(time_from_string(ts));
    time_duration td = tt - tref;
	
	boost::this_thread::interruption_point();

	try
	{
		boost::asio::io_service io_service;
		tcp::resolver resolver(io_service);
		tcp::resolver::query query(tcp::v4(), "localhost", "7879");
		tcp::resolver::iterator iterator = resolver.resolve(query);
		
		boost::asio::deadline_timer t(io_service);
	
		const int bucketSize = 64;
		const int imageSizeX = resolutionX();
		const int imageSizeY = resolutionY();
		int rect[4];
				
		for(int by = 0; by <= imageSizeY/bucketSize; by++) {
			rect[2] = by * bucketSize;
			rect[3] = rect[2] + bucketSize - 1;
			if(rect[3] > imageSizeY - 1) rect[3] = imageSizeY - 1;
			for(int bx = 0; bx <= imageSizeX/bucketSize; bx++) {
				rect[0] = bx * bucketSize;
				rect[1] = rect[0] + bucketSize - 1;
				if(rect[1] > imageSizeX - 1) rect[1] = imageSizeX - 1;
				
				const float grey = (float)((rand() + td.seconds() * 391) % 457) / 457.f;
				const unsigned npix = (rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1);
				int npackage = npix * 16 / 4096;
				if((npix * 16) % 4096 > 0) npackage++;
				tcp::socket s(io_service);
				s.connect(*iterator);
				boost::asio::write(s, boost::asio::buffer((char *)rect, 16));
				boost::array<char, 128> buf;
				boost::system::error_code error;
				size_t reply_length = s.read_some(boost::asio::buffer(buf), error);

				//std::cout<<" bucket("<<rect[0]<<","<<rect[1]<<","<<rect[2]<<","<<rect[3]<<")\n";
				
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
				
				s.close();
				
				boost::this_thread::interruption_point();
				
				t.expires_from_now(boost::posix_time::seconds(0.5));
				t.wait();
			}
		}

	}
	catch (std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << "\n";
	}
	
}