/*
 *  DirectionalBendAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DirectionalBendAttribs.h"
#include <geom/ATriangleMesh.h>
#include <geom/DirectionalBendDeformer.h>
#include <smp/GeodesicSphere.h>
#include <math/miscfuncs.h>
#include <gar_common.h>

using namespace aphid;

int DirectionalBendAttribs::sNumInstances = 0;

DirectionalBendAttribs::DirectionalBendAttribs() : PieceAttrib(gar::gtDirectionalVariant),
m_inAttr(NULL),
m_inGeom(NULL),
m_exclR(1.f)
{
    m_instId = sNumInstances;
	sNumInstances++;
	
	addSplineAttrib(gar::nBendVariation);
	addSplineAttrib(gar::nNoiseVariation);
	
	m_dfm = new DirectionalBendDeformer;
	m_samples = new smp::GeodesicSphere;
	for(int i=0;i<36;++i) 
	    m_outGeom[i] = new ATriangleMesh;
}

void DirectionalBendAttribs::setInputGeom(ATriangleMesh* x)
{ m_inGeom = x; }

bool DirectionalBendAttribs::hasGeom() const
{ return m_inGeom != NULL; }

int DirectionalBendAttribs::numGeomVariations() const
{ return 36; }

ATriangleMesh* DirectionalBendAttribs::selectGeom(gar::SelectProfile* prof) const
{
	if(prof->_condition != gar::slIndex)
		prof->_index = rand() % numGeomVariations();
		
    prof->_exclR = m_exclR;
	prof->_height = m_geomHeight;
	return m_outGeom[prof->_index];
}

bool DirectionalBendAttribs::update()
{    
    if(!m_inAttr)
        return false;
    
	gar::SelectProfile selprof;
    m_inGeom = m_inAttr->selectGeom(&selprof);
    
    if(!m_inGeom)
        return false;
		
	m_exclR = selprof._exclR;
	m_geomHeight = selprof._height;
		
	SplineMap1D* ls = m_dfm->bendSpline();
	SplineMap1D* ns = m_dfm->noiseSpline();
	
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nBendVariation);
	gar::SplineAttrib* ans = (gar::SplineAttrib*)findAttrib(gar::nNoiseVariation);
	
	updateSplineValues(ls, als);
	updateSplineValues(ns, ans);
    
	m_dfm->computeRowWeight(m_inGeom);
	m_samples->generateSamples();
	
	for(int i=0;i<36;++i) {
        m_dfm->setDirection(m_samples->getSample(i, 0.3f) );
        m_dfm->deform(m_inGeom);
		m_dfm->updateGeom(m_outGeom[i], m_inGeom);
	}
	
	computeTexcoord(m_outGeom, 36, m_inAttr->texcoordBlockAspectRatio() );
	
	return true;
}

int DirectionalBendAttribs::attribInstanceId() const
{ return m_instId; }

void DirectionalBendAttribs::connectTo(PieceAttrib* another, const std::string& portName)
{
    if(!another->hasGeom()) {
        std::cout<<"\n ERROR DirectionalBendAttribs cannot connect input geom ";
        m_inGeom = NULL;  
        return;
    }
    
    m_inAttr = another;
    update();
}

bool DirectionalBendAttribs::isGeomStem() const
{
	if(!m_inAttr)
		return false;
	return m_inAttr->isGeomStem();
}

bool DirectionalBendAttribs::isGeomLeaf() const
{
	if(!m_inAttr)
		return false;
	return m_inAttr->isGeomLeaf();
}

bool DirectionalBendAttribs::canConnectToViaPort(const PieceAttrib* another, const std::string& portName) const
{
	return (another->isGeomStem() || another->isGeomLeaf() );
}
