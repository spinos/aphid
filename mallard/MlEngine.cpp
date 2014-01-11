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
#include <AdaptableStripeBuffer.h>
#include <boost/format.hpp>
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
    loadPlugin("mtoa_shaders.dll");
	
	const AtNodeEntry* nodeEntry = AiNodeEntryLookUp("userDataColor");
	if(nodeEntry == NULL) std::clog<<"\nWARNING: userDataColor node entry doesn't exist! Most likely mtoa_shaders.all is not loaded.\n";
    
    AtNode* options = AiNode("options");
    AtArray* outputs  = AiArrayAllocate(1, 1, AI_TYPE_STRING);
    AiArraySetStr(outputs, 0, "RGBA RGBA output:gaussian_filter output/foo");
    AiNodeSetArray(options, "outputs", outputs);
    
    AiNodeSetInt(options, "xres", resolutionX());
    AiNodeSetInt(options, "yres", resolutionY());
    AiNodeSetInt(options, "AA_samples", 5);
    AiNodeSetInt(options, "GI_diffuse_samples", 3);
	
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
    AiNodeSetFlt(sphere, "radius", 1.f);
    AiM4Identity(matrix);
    matrix[3][0] = 0.f;
    matrix[3][1] = 0.f;
    matrix[3][2] = 50.f;
    AiNodeSetMatrix(sphere, "matrix", matrix);
    
    translateLights();
    translateCurves();

    logRenderError(AiRender(AI_RENDER_MODE_CAMERA));
    
    AiEnd();
#endif
}

void MlEngine::testOutput()
{
	if(!m_barb) return;
	
	translateCurves();
	
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

void MlEngine::translateCurves()
{
    const unsigned n = m_barb->numBlocks();
    if(n < 1) return;
    for(unsigned i = 0; i < n; i++) {
        translateBlock(m_barb->block(0));
        m_barb->clearBlock(0);
    }
}

void MlEngine::translateBlock(AdaptableStripeBuffer * src)
{
	const unsigned ns = src->numStripe();
	const unsigned np = src->numPoints();
	const std::string curveName = (boost::format("/obj/curve%1%") % rand()).str();
#ifdef WIN32
    AtNode *curveNode = AiNode("curves");

    AiNodeSetStr(curveNode, "name", curveName.c_str());
    AiNodeSetStr(curveNode, "basis", "catmull-rom");
    //AiNodeSetStr(curveNode, "basis", "linear");
    //AiNodeSetStr(curveNode, "basis", "bezier");
    //AiNodeSetInt(curveNode, "sidedness", 2);
    AiNodeSetInt(curveNode, "visibility", 65523);
    AiNodeSetFlt(curveNode, "min_pixel_width", .5f);
    AiNodeSetBool(curveNode, "opaque", false);
    AiNodeSetInt(curveNode, "max_subdivs", 3);
    AtArray* counts = AiArrayAllocate(ns, 1, AI_TYPE_UINT);
#endif    
    unsigned * ncv = src->numCvs();
    unsigned npt = 0;
    unsigned nw = 0;
	for(unsigned i = 0; i < ns; i++) {
#ifdef WIN32
	    AiArraySetUInt(counts, i, ncv[i] + 2);
#endif
	    npt += ncv[i] + 2;
	    nw += ncv[i];
	}
#ifdef WIN32

    AtArray* points = AiArrayAllocate(npt, 1, AI_TYPE_POINT);
    AtArray* radius = AiArrayAllocate(nw, 1, AI_TYPE_FLOAT);
	
    Vector3F * pos = src->pos();
	Vector3F * col = src->col();
	float * w = src->width();
	
	AtPoint sample;
	
	unsigned asrc = 0;
	unsigned ap = 0;
	unsigned aw = 0;
	for(unsigned j = 0; j < ns; j++) {
	    for(unsigned i = 0; i < ncv[j]; i++) {
	        
            sample.x = pos[asrc].x;
            sample.y = pos[asrc].y;
            sample.z = pos[asrc].z;
           
            AiArraySetPnt(points, ap, sample);
            ap++;
            
            AiArraySetFlt(radius, aw, w[asrc]);
            aw++;
            
            
            if(i==0 || i== ncv[j] - 1) {
                AiArraySetPnt(points, ap, sample);
                ap++;
           }
            
            asrc++;
        }
	}
	std::clog<<"asrc "<<asrc<<" np "<<np<<"\n";
	
	AiNodeSetArray(curveNode, "num_points", counts);
	AiNodeSetArray(curveNode, "points", points);
	AiNodeSetArray(curveNode, "radius", radius);
	
	AiNodeDeclare(curveNode, "colors", "varying RGB");
	AtArray* colors = AiArrayAllocate(np, 1, AI_TYPE_RGB);
	
	AtRGB acol;
	for(unsigned i = 0; i < np; i++) {
	    acol.r = col[i].x;
	    acol.g = col[i].y;
	    acol.b = col[i].z;
	    AiArraySetRGB(colors, i, acol);
	}
	
	AiNodeSetArray(curveNode, "colors", colors);
	
	AtNode * usrCol = AiNode("userDataColor");
	AiNodeSetStr(usrCol, "colorAttrName", "colors");
	
	AtNode *hair = AiNode("hair");
	AiNodeSetFlt(hair, "ambdiff", 1.f);
	AiNodeSetFlt(hair, "spec", 1.f);
	AiNodeSetFlt(hair, "spec2", 2.f);
	//AiNodeSetRGB(hair, "rootcolor", 0.2f, 0.2f, 0.2f);
	//AiNodeSetRGB(hair, "tipcolor", 0.1f, 0.1f, 0.1f);
	if(AiNodeLink(usrCol, "rootcolor", hair)) std::clog<<"linked";
	if(AiNodeLink(usrCol, "tipcolor", hair)) std::clog<<"linked";
	if(AiNodeLink(usrCol, "spec_color", hair)) std::clog<<"linked";
	if(AiNodeLink(usrCol, "spec2_color", hair)) std::clog<<"linked";
	//AtNode *hair = AiNode("utility");
	//if(AiNodeLink(usrCol, "color", hair)) std::clog<<"linked";

	AiNodeSetPtr(curveNode, "shader", hair);
#endif
	std::clog<<curveName<<" n curves "<<ns<<" n points "<<np<<"\n";
}

void MlEngine::translateLights()
{
	LightGroup * g = lights();
	for(unsigned i = 0; i < g->numLights(); i++) 
		translateLight(g->getLight(i));
}

void MlEngine::translateLight(BaseLight * l)
{
	switch (l->entityType()) {
		case TypedEntity::TDistantLight:
			translateDistantLight(static_cast<DistantLight *>(l));
			break;
		case TypedEntity::TPointLight:
			translatePointLight(static_cast<PointLight *>(l));
			break;
		case TypedEntity::TSquareLight:
			translateSquareLight(static_cast<SquareLight *>(l));
			break;
		default:
			break;
	}
}

void MlEngine::translateDistantLight(DistantLight * l)
{
#ifdef WIN32
	AtNode * light = AiNode("distant_light");
    //AiNodeSetInt(light, "samples", 3);
    AiNodeSetStr(light, "name", l->name().c_str());
    AiNodeSetFlt(light, "intensity", l->intensity());
    AtMatrix matrix;
	setMatrix(l->worldSpace(), matrix);
    AiNodeSetMatrix(light, "matrix", matrix);
#endif
}

void MlEngine::translatePointLight(PointLight * l)
{
#ifdef WIN32
	AtNode * light = AiNode("point_light");
    AiNodeSetStr(light, "name", l->name().c_str());
    AiNodeSetFlt(light, "intensity", l->intensity());
    AtMatrix matrix;
	setMatrix(l->worldSpace(), matrix);
    AiNodeSetMatrix(light, "matrix", matrix);
#endif
}

void MlEngine::translateSquareLight(SquareLight * l)
{

}
