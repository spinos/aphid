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
#include <BaseCamera.h>
#include <boost/asio.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time;
using boost::asio::ip::tcp;
#define PACKAGESIZE 1024
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
	//interruptRender();
}

void MlEngine::setWorks(BarbWorks * w)
{
	m_barb = w;
}

void MlEngine::preRender() 
{
	interruptRender();
	m_barb->setEyePosition(camera()->eyePosition());
	m_barb->setFieldOfView(camera()->fieldOfView());
	m_barb->clearBarbBuffer();
	m_workingThread = boost::thread(boost::bind(&BarbWorks::createBarbBuffer, this->m_barb, this));
	m_progressingThread = boost::thread(boost::bind(&MlEngine::monitorProgressing, this, this->m_barb));
}

void MlEngine::render()
{
    AnorldFunc::render();
#ifdef WIN32
    m_workingThread = boost::thread(boost::bind(&MlEngine::fineOutput, this));
#else
	m_workingThread = boost::thread(boost::bind(&MlEngine::testOutput, this));
#endif
}

void MlEngine::interruptRender()
{
	std::cout<<" interrupted ";
	m_workingThread.interrupt();
	m_progressingThread.interrupt();
}

void MlEngine::fineOutput()
{
    ptime tt(second_clock::local_time());
	std::cout<<"fine output begins at "<<to_simple_string(tt)<<"\n";
#ifdef WIN32
	AiBegin();
    loadPlugin("./driver_foo.dll");
    
    AtNode* options = AiNode("options");
    AtArray* outputs  = AiArrayAllocate(1, 1, AI_TYPE_STRING);
    AiArraySetStr(outputs, 0, "RGBA RGBA output:gaussian_filter output/foo");
    AiNodeSetArray(options, "outputs", outputs);
    
    AiNodeSetInt(options, "xres", resolutionX());
    AiNodeSetInt(options, "yres", resolutionY());
    AiNodeSetInt(options, "AA_samples", 3);
	
    AtNode* driver = AiNode("driver_foo");
    AiNodeSetStr(driver, "name", "output/foo");
    
    AtNode * acamera = AiNode("persp_camera");
    AiNodeSetStr(acamera, "name", "/obj/cam");
    AiNodeSetFlt(acamera, "fov", camera()->fieldOfView());
    AiNodeSetFlt(acamera, "near_clip", camera()->nearClipPlane());
    AiNodeSetFlt(acamera, "far_clip", camera()->farClipPlane());
    
    AtMatrix matrix;
    setMatrix(camera()->fSpace, matrix);
	AiNodeSetMatrix(acamera, "matrix", matrix);

	AiNodeSetPtr(options, "camera", acamera);
    
	AtNode * filter = AiNode("gaussian_filter");
    AiNodeSetStr(filter, "name", "output:gaussian_filter");

    AtNode * standard = AiNode("standard");
    AiNodeSetStr(standard, "name", "/shop/standard1");
    AiNodeSetRGB(standard, "Kd_color", 1, 0, 0);

    AtNode * sphere = AiNode("sphere");
    AiNodeSetPtr(sphere, "shader", standard);
    
    AtNode * light = AiNode("point_light");
    AiNodeSetStr(light, "name", "/obj/lit");
    AiNodeSetFlt(light, "intensity", 1024);
    AiM4Identity(matrix);
    matrix[3][0] = -10.f;
    AiNodeSetMatrix(light, "matrix", matrix);

    logRenderError(AiRender(AI_RENDER_MODE_CAMERA));
    
    AiEnd();
#endif
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
	
	char dataPackage[PACKAGESIZE];

	try
	{
		boost::asio::io_service io_service;
		tcp::resolver resolver(io_service);
		tcp::resolver::query query(tcp::v4(), "localhost", "7879");
		tcp::resolver::iterator iterator = resolver.resolve(query);
		tcp::socket s(io_service);
		
		boost::asio::deadline_timer t(io_service);
	
		const int bucketSize = 64;
		const int imageSizeX = resolutionX();
		const int imageSizeY = resolutionY();
				
		for(int by = 0; by <= imageSizeY/bucketSize; by++) {
			for(int bx = 0; bx <= imageSizeX/bucketSize; bx++) {
				int * rect = (int *)dataPackage;
				
				rect[2] = by * bucketSize;
				rect[3] = rect[2] + bucketSize - 1;
				if(rect[3] > imageSizeY - 1) rect[3] = imageSizeY - 1;
			
				rect[0] = bx * bucketSize;
				rect[1] = rect[0] + bucketSize - 1;
				if(rect[1] > imageSizeX - 1) rect[1] = imageSizeX - 1;
				
				const float grey = (float)((rand() + td.seconds() * 391) % 457) / 457.f;
				const unsigned npix = (rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1);
				int npackage = npix * 16 / PACKAGESIZE;
				if((npix * 16) % PACKAGESIZE > 0) npackage++;
				
				s.connect(*iterator);
		
				boost::asio::write(s, boost::asio::buffer(dataPackage, 16));
				//std::cout<<"sent    bucket("<<rect[0]<<","<<rect[1]<<","<<rect[2]<<","<<rect[3]<<")\n";
				
				boost::array<char, 32> buf;
				boost::system::error_code error;
				
				size_t reply_length = s.read_some(boost::asio::buffer(buf), error);
				
				float *color = (float *)dataPackage;
				for(int i = 0; i < PACKAGESIZE / 16; i++) {
					color[i * 4] = color[i * 4 + 1] = color[i * 4 + 2] = grey;
					color[i * 4 + 3] = 1.f;
				}
					
				for(int i=0; i < npackage; i++) {
					boost::asio::write(s, boost::asio::buffer(dataPackage, PACKAGESIZE));
					reply_length = s.read_some(boost::asio::buffer(buf), error);
				}
				
				boost::asio::write(s, boost::asio::buffer(dataPackage, 32));
				reply_length = s.read_some(boost::asio::buffer(buf), error);

				s.close();
				//t.expires_from_now(boost::posix_time::seconds(1));
				//t.wait();
				
				boost::this_thread::interruption_point();
			}
		}
		
	}
	catch (std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << "\n";
	}
	
}

void MlEngine::monitorProgressing(BarbWorks * work)
{
	boost::asio::io_service io_service;
	boost::asio::deadline_timer t(io_service);
	for(;;) {
		t.expires_from_now(boost::posix_time::seconds(5));
		t.wait();
		boost::this_thread::interruption_point();
		std::clog<<" "<<work->percentFinished() * 100<<"% ";
		if(work->percentFinished() == 1.f) break;
	}
}