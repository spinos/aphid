/*
 *  MlEngine.h
 *  mallard
 *
 *  Created by jian zhang on 12/31/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <RenderEngine.h>
#include <boost/thread.hpp>
class BarbWorks;
class MlEngine : public RenderEngine {
public:
	MlEngine();
	MlEngine(BarbWorks * w);
	virtual ~MlEngine();
	void setWorks(BarbWorks * w);
	virtual void render();
	void interruptRender();
protected:

private:
	void processBarbs();
	void testOutput();
	
private:
	boost::thread m_workingThread;
	BarbWorks * m_barb;
};