/*
 *  ColorSampler.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ColorSampler.h"
#include <maya/MFloatPointArray.h>
#include <maya/MFloatArray.h>
#include <maya/MFloatMatrix.h>
#include <maya/MItMeshPolygon.h>
#include <mama/AHelper.h>
#include <mama/ASearchHelper.h>
#include <mama/AttributeHelper.h>
#include <geom/ConvexShape.h>
#include <geom/ATriangleMesh.h>
#include <sdb/VectorArray.h>
#include <geom/ConvexShape.h>
#include <img/ExrImage.h>

namespace aphid {

ColorSampler::ColorSampler()
{}

void ColorSampler::SampleMeshTrianglesInGroup(sdb::VectorArray<cvx::Triangle> & tris,
								BoundingBox & bbox,
							const MDagPath & groupPath)
{
    MDagPathArray meshPaths;
	ASearchHelper::LsAllTypedPaths(meshPaths, groupPath, MFn::kMesh);
	const int n = meshPaths.length();
	if(n < 1) {
		AHelper::Info<MString>("find no mesh in group", groupPath.fullPathName() );
		return;
	}
	
	SampleProfile profile;
	profile.m_imageSampler = NULL;
	profile.m_defaultColor.set(.2f, .53f, .17f);
	
	for(int i=0;i<n;++i) {
		const int triBegin = tris.size();
		const MDagPath & curMeshP = meshPaths[i];
	    GetMeshTriangles(tris, bbox, curMeshP, groupPath );
		const int triEnd = tris.size();
		MObject curMeshO = curMeshP.node();
		
		GetMeshDefaultColor(&profile, curMeshO);
		GetMeshImageFileName(&profile, curMeshO);
		
		SampleTriangles(tris, triBegin, triEnd, &profile);
	}
	
	if(profile.m_imageSampler) {
		delete profile.m_imageSampler;
	}
	
}

void ColorSampler::GetMeshDefaultColor(SampleProfile * profile,
						const MObject & node)
{
	MObject attr;
	if(!AttributeHelper::HasNamedAttribute(attr, node, "pxpnColor") ) {
		return;
	}
	
	double r, g, b;
	MFnDependencyNode fnode(node);
	AttributeHelper::getColorAttributeByName(fnode, "pxpnColor", r, g, b);
	
	profile->m_defaultColor.set(r, g, b);
	std::cout<<"\n ColorSampler::GetMeshDefaultColor "<<profile->m_defaultColor;
}

void ColorSampler::GetMeshImageFileName(SampleProfile * profile,
						const MObject & node)
{
	MObject attr;
	if(!AttributeHelper::HasNamedAttribute(attr, node, "pxpnImage") ) {
		return;
	}
	
	MString msimg;
	AttributeHelper::getStringAttributeByName(node, "pxpnImage", msimg);
	
	if(msimg.length() < 5) {
		if(profile->m_imageSampler) {
			delete profile->m_imageSampler;
			profile->m_imageSampler = NULL;
		}
		return;
	}
	
	std::cout<<"\n ColorSampler::GetMeshImageFileName "<<msimg.asChar();
	
	if(!profile->m_imageSampler) {
		profile->m_imageSampler = new ExrImage;
	}
	
	const std::string imgname(msimg.asChar());
	
	profile->m_imageSampler->read(imgname);
	profile->m_imageSampler->verbose();
	
}

bool ColorSampler::SampleTriangles(sdb::VectorArray<cvx::Triangle> & tris,
						const int & iBegin, const int & iEnd,
						SampleProfile * profile)
{
	const int numSamples = iEnd - iBegin;
	std::cout<<"\n ColorSampler::SampleTriangles n tri "<<numSamples;
	
	bool doSampleImage = false;
	if(profile->m_imageSampler) {
		doSampleImage = profile->m_imageSampler->isValid();
	}
	
	Float2 auv;
	const float uvContribs[3] = {.333f, .333f, .333f};
	Vector3F vc = profile->m_defaultColor;
	
	for(int i=iBegin;i<iEnd;++i) {
		cvx::Triangle * atri = tris[i];
		atri->resetNC();
		
		if(doSampleImage) {
			auv = atri->interpolateTexcoord(uvContribs);
			profile->m_imageSampler->sample(auv.x, auv.y, 3, (float *)&vc);
		}
		
		for(int j=0;j<3;++j) {
			atri->setC(vc, j);
		}
		
	}
	
	return true;
}

}