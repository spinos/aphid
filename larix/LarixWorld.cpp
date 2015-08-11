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
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"

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
    m_cacheFile = new H5FieldOut;
    if(!m_cacheFile->create("./dposition.tmp")) 
        std::cout<<"\n error: larix world cannot create dposition field cache file!\n";
    m_isCachingFinished = false;
}

LarixWorld::~LarixWorld() 
{ 
    if(m_cloud) delete m_cloud; 
    if(m_mesh) delete m_mesh;
	if(m_field) delete m_field;
	delete m_sourceFile;
    delete m_cacheFile;
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

void LarixWorld::setCacheRange()
{ 
    if(!m_sourceFile->isOpened()) return;
    m_sourceFile->readFrameRange(); 
    m_sourceFile->verbose();
}

void LarixWorld::beginCache()
{
    setCacheRange();
    if(m_cacheFile->isOpened()) {
        m_cacheFile->writeFrameRange(m_sourceFile);
        m_cacheFile->addField("lax", m_field);
    }
    m_sourceFile->frameBegin();
}

int LarixWorld::currentCacheFrame() const
{ return m_sourceFile->currentFrame(); }

void LarixWorld::progressFrame()
{
    if(m_sourceFile->isOutOfRange()) {
        m_isCachingFinished = true;
        return;
    }
    
    if(m_sourceFile->isOpened()) {
        m_sourceFile->readFrame();
        *m_sourceP -= m_mesh->pointsBuf();
        m_field->computeChannelValue("dP", m_sourceP, m_mesh->sampler());
    }
    
    if(m_cacheFile->isOpened()) 
        m_cacheFile->writeFrame(m_sourceFile->currentFrame());
    
    m_sourceFile->nextFrame(); 
}

bool LarixWorld::setFileOut(const std::string & fileName)
{
    if(!isCachingFinished()) return false;
    m_cacheFile->verbose();
// flush b4 cpy
    m_cacheFile->flush();
    boost::filesystem::path fromtFp("./dposition.tmp");
    boost::filesystem::path toFp(fileName);
    try 
    {
    boost::filesystem::copy_file(fromtFp, toFp,
             boost::filesystem::copy_option::overwrite_if_exists);
    }
    catch (const boost::filesystem::filesystem_error& ex)
    {
        std::cout << "\n" << ex.what() << '\n';
        return false;
    }
// test resulting file
    H5FieldIn t;
    if(!t.open(fileName)) {
        std::cout<<"\n cannot open destination file";
        return false;
    }
    
    AdaptiveField * fld = (AdaptiveField *)t.fieldByIndex(0);
    
    std::cout<<"\n adaptive field n cells "<< fld->numCells();
    t.close();
	return true;
}

bool LarixWorld::isCachingFinished() const
{ return m_isCachingFinished && m_cacheFile->isOpened(); }
//:~