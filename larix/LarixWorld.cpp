/*
 *  LarixWorld.cpp
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "LarixWorld.h"
#include <ATetrahedronMesh.h>
#include <APointCloud.h>
#include <AdaptiveField.h>
#include <H5FieldIn.h>
#include <H5FieldOut.h>

LarixWorld::LarixWorld() 
{ 
    m_sourceP = 0;
    m_cloud = 0;
    m_mesh = 0;
	m_field = 0;
	m_sourceFile = new H5FieldIn;
	if(!m_sourceFile->open("./position.tmp"))
		std::cout<<"\n error: larix world cannot open position cache file!\n";
 
    checkSourceField();
	m_outFile = new H5FieldOut;
}

LarixWorld::~LarixWorld() 
{ 
    if(m_cloud) delete m_cloud; 
    if(m_mesh) delete m_mesh;
	if(m_field) delete m_field;
	delete m_sourceFile;
	delete m_outFile;
}

void LarixWorld::setTetrahedronMesh(ATetrahedronMesh * m)
{ m_mesh = m; }

ATetrahedronMesh * LarixWorld::tetrahedronMesh() const
{ return m_mesh; }

void LarixWorld::setPointCloud(APointCloud * pc)
{ m_cloud = pc; }

APointCloud * LarixWorld::pointCloud() const
{ return m_cloud; }

void LarixWorld::setField(AdaptiveField * g)
{ m_field = g; }

AdaptiveField * LarixWorld::field() const
{ return m_field; }

bool LarixWorld::checkSourceField()
{
    if(m_sourceFile->numFields() < 1) {
        std::cout<<"\n source has no fields ";
        return false;
    }
    
    AField * f = m_sourceFile->fieldByIndex(0);
    
    TypedBuffer * pb = f->namedChannel("P");
    
    if(!pb) {
        std::cout<<"\n field has no P channel ";
        return false;
    }
    std::cout<<"\n P channel n elm "<<pb->numElements();
    m_sourceP = pb;
    return true;
}

bool LarixWorld::hasSourceP() const
{ return m_sourceP != 0;}

TypedBuffer * LarixWorld::sourceP()
{ return m_sourceP; }

void LarixWorld::setSourceP(TypedBuffer * x)
{ m_sourceP = x; }

int LarixWorld::cacheRangeMin() const
{ return m_sourceFile->FirstFrame; }

int LarixWorld::cacheRangeMax() const
{ return m_sourceFile->LastFrame; }

void LarixWorld::readCacheRange()
{ if(m_sourceFile->isOpened()) m_sourceFile->readFrameRange(); }

void LarixWorld::beginCache()
{
    readCacheRange();
    m_sourceFile->frameBegin();
}

int LarixWorld::currentCacheFrame() const
{ return m_sourceFile->currentFrame(); }

void LarixWorld::progressFrame()
{ 
    if(!m_sourceFile->isOpened()) return;
    if(m_sourceFile->isOutOfRange()) return;
    m_sourceFile->readFrame();
    *m_sourceP -= m_mesh->pointsBuf();
    m_field->computeChannelValue("dP", m_sourceP, m_mesh->sampler());
    //m_outColor->copyFrom(m_field->namedChannel("dP")->data());
	if(m_outFile->isOpened()) 
		m_outFile->writeFrame(m_sourceFile->currentFrame());
	
    m_sourceFile->nextFrame(); 
}

bool LarixWorld::setFileOut(const std::string & fileName)
{
	if(!m_outFile->create(fileName)) {
		std::cout<<"\n error: larix world cannot create out cache file "
			<<fileName;
		return false;
	}
 
	std::cout<<"\n larix world begin cache frames ("<<m_sourceFile->FirstFrame
    <<","<<m_sourceFile->LastFrame<<")";
    
	m_outFile->writeFrameRange(m_sourceFile);
	m_outFile->addField("lax", m_field);
	beginCache();
	return true;
}

void LarixWorld::setCellColorOutBuf(BaseBuffer * b)
{ m_outColor = b; }
//:~