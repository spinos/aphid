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

using namespace aphid;

int DirectionalBendAttribs::sNumInstances = 0;

DirectionalBendAttribs::DirectionalBendAttribs() :
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

ATriangleMesh* DirectionalBendAttribs::selectGeom(int x, float& exclR) const
{
    exclR = m_exclR;
	return m_outGeom[x]; 
}

bool DirectionalBendAttribs::update()
{    
    if(!m_inAttr)
        return false;
    
    m_inGeom = m_inAttr->selectGeom(0, m_exclR);
    
    if(!m_inGeom)
        return false;
		
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

void DirectionalBendAttribs::connectTo(PieceAttrib* another)
{
    if(!another->hasGeom()) {
        std::cout<<"\n ERROR DirectionalBendAttribs cannot connect input geom ";
        m_inGeom = NULL;  
        return;
    }
    
    m_inAttr = another;
    update();
}
