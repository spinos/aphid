/*
 *  HesperisFile.cpp
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisFile.h"
#include <geom/CurveGroup.h>
#include <math/BaseBuffer.h>
#include <geom/APolygonalMesh.h>
#include <geom/ATriangleMeshGroup.h>
#include <geom/ATetrahedronMeshGroup.h>
#include <BaseTransform.h>
#include <foundation/AAttribute.h>
#include <geom/GeometryArray.h>
#include <foundation/SHelper.h>
#include <sstream>
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>

namespace aphid {

AFrameRange HesperisFile::Frames;
bool HesperisFile::DoStripNamespace = true;

HesperisFile::HesperisFile() {}
HesperisFile::HesperisFile(const char * name) : HFile(name) 
{
	m_readComp = RNone;
	m_writeComp = WCurve;
}

HesperisFile::~HesperisFile() {}

void HesperisFile::setReadComponent(ReadComponent comp)
{ m_readComp = comp; }

void HesperisFile::setWriteComponent(WriteComponent comp)
{ m_writeComp = comp; }

bool HesperisFile::doWrite(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseFile::FileNotWritable);
		return false;
	}
	
	HWorld grpWorld;
	grpWorld.save();
	
	switch (m_writeComp) {
		case WCurve:
			writeCurve();
			break;
		case WTetra:
			writeTetrahedron();
			break;
		case WTri:
			writeTriangle();
			break;
		case WTransform:
			writeTransform();
			break;
        case WPoly:
			writePolygon();
			break;
		case WAttrib:
			writeAttribute();
			break;
	    case WBundle:
	        writeBundle();
		default:
			break;
	}
	
	writeFrames();
	grpWorld.close();
	HObject::FileIO.close();
	
	std::cout<<" finished writing hesperis file at "<<grpWorld.modifiedTimeStr()<<"\n";
	
	return true;
}

bool HesperisFile::writeTransform()
{
	std::map<std::string, BaseTransform *>::iterator ittrans = m_transforms.begin();
	for(; ittrans != m_transforms.end(); ++ittrans) {
		//std::cout<<" write transform "<<WorldPath(ittrans->first)
        //<<"\n";
		HTransform grp(WorldPath(ittrans->first));
		grp.save(ittrans->second);
		grp.close();
	}
	return true;
}

bool HesperisFile::writeCurve()
{
	std::map<std::string, CurveGroup *>::iterator itcurve = m_curves.begin();
	for(; itcurve != m_curves.end(); ++itcurve) {
		//std::cout<<" write curve "<<WorldPath(itcurve->first)
        //<<"\n";
		HCurveGroup grpCurve(WorldPath(itcurve->first));
		grpCurve.save(itcurve->second);
		grpCurve.close();
	}
	return true;
}

bool HesperisFile::writeTetrahedron()
{
	std::map<std::string, ATetrahedronMeshGroup *>::iterator it = m_terahedrons.begin();
	for(; it != m_terahedrons.end(); ++it) {
		//std::cout<<" write tetrahedron mesh "<<WorldPath(it->first)
		//<<"\n";
		HTetrahedronMeshGroup grp(WorldPath(it->first));
		grp.save(it->second);
		grp.close();
	}
	return true;
}

bool HesperisFile::writeTriangle()
{
	std::map<std::string, ATriangleMeshGroup *>::iterator it = m_triangleMeshes.begin();
	for(; it != m_triangleMeshes.end(); ++it) {
		//std::cout<<" write triangle mesh "<<WorldPath(it->first)
		//<<"\n";
		HTriangleMeshGroup grp(WorldPath(it->first));
		grp.save(it->second);
		grp.close();
	}
	return true;
}

bool HesperisFile::writePolygon()
{
    std::map<std::string, APolygonalMesh *>::iterator it = m_polyMeshes.begin();
	for(; it != m_polyMeshes.end(); ++it) {
		//std::cout<<" write poly mesh "<<WorldPath(it->first)
		//<<"\n";
		HPolygonalMesh grp(WorldPath(it->first));
		grp.save(it->second);
		grp.close();
	}
	return true;
}

bool HesperisFile::writeAttribute()
{
	std::map<std::string, AAttribute *>::iterator it = m_attribs.begin();
	for(; it != m_attribs.end(); ++it) {
		//std::cout<<" write attrib "<<WorldPath(it->first)
		//<<"\n";
		HAttributeGroup grp(WorldPath(it->first));
		grp.save(it->second);
		grp.close();
	}
	return true;
}

void HesperisFile::addAttribute(const std::string & name, AAttribute * data)
{ m_attribs[checkPath(name)] = data; }

void HesperisFile::addTransform(const std::string & name, BaseTransform * data)
{ m_transforms[checkPath(name)] = data; }

void HesperisFile::addCurve(const std::string & name, CurveGroup * data)
{ m_curves[checkPath(name)] = data; }

void HesperisFile::addTetrahedron(const std::string & name, ATetrahedronMeshGroup * data)
{ m_terahedrons[checkPath(name)] = data; }

void HesperisFile::addTriangleMesh(const std::string & name, ATriangleMeshGroup * data)
{ m_triangleMeshes[checkPath(name)] = data; }

void HesperisFile::addPolygonalMesh(const std::string & name, APolygonalMesh * data)
{ m_polyMeshes[checkPath(name)] = data; }

bool HesperisFile::doRead(const std::string & fileName)
{
	if(!HFile::doRead(fileName)) return false;
	
	std::cout<<" reading hesperis file "<<fileName<<"\n";
	HWorld grpWorld;
	grpWorld.load();
	
	switch (m_readComp) {
		case RCurve:
			listCurve(&grpWorld);
			readCurve();
			break;
		case RTetra:
            listTetrahedron(&grpWorld);
			readTetrahedron();
			break;
		case RTri:
            listTriangle(&grpWorld);
			readTriangle();
		default:
			break;
	}
	
	readFrames(&grpWorld);
	grpWorld.close();
	
	std::cout<<" finished reading hesperis file modified at "<<grpWorld.modifiedTimeStr()<<"\n";
	
	return true;
}

bool HesperisFile::listCurve(HBase * grp)
{
	clearCurves();
    std::vector<std::string > curveNames;
    LsNames<HCurveGroup>(curveNames, grp);
	
	std::vector<std::string>::const_iterator it = curveNames.begin();
	for(;it!=curveNames.end();++it)
		addCurve(*it, new CurveGroup);

	return true;
}

bool HesperisFile::readCurve()
{
	bool allValid = true;
	std::map<std::string, CurveGroup *>::iterator itcurve = m_curves.begin();
	for(; itcurve != m_curves.end(); ++itcurve) {
		std::cout<<" read curve "<<itcurve->first
		<<"\n";
		HCurveGroup grpCurve(itcurve->first);
		if(!grpCurve.load(itcurve->second)) {
			std::cout<<" cannot load "<<itcurve->first
			<<"\n";
			allValid = false;
		}
		
		grpCurve.close();
	}
	
	if(!allValid)
		std::cout<<" encounter problem(s) reading curves.\n";

	return allValid;
}

bool HesperisFile::listTetrahedron(HBase * grp)
{
    clearTetrahedronMeshes();
    std::vector<std::string > tetraNames;
    LsNames<HTetrahedronMeshGroup>(tetraNames, grp);
	
	std::vector<std::string>::const_iterator it = tetraNames.begin();
	for(;it!=tetraNames.end();++it)
		addTetrahedron(*it, new ATetrahedronMeshGroup);

	return true;
}

bool HesperisFile::readTetrahedron()
{
    bool allValid = true;
	std::map<std::string, ATetrahedronMeshGroup *>::iterator it = m_terahedrons.begin();
	for(; it != m_terahedrons.end(); ++it) {
		std::cout<<" read tetrahedron mesh "<<it->first<<"\n";
		HTetrahedronMeshGroup grp(it->first);
		if(!grp.load(it->second)) {
			std::cout<<" cannot load "<<it->first;
			allValid = false;
		}
		grp.close();
	}
	
	if(!allValid)
		std::cout<<" encounter problem(s) reading tetrahedrons.\n";

	return allValid;
}

bool HesperisFile::listTriangle(HBase * grp)
{
    clearTriangleMeshes();
    std::vector<std::string > triNames;
    LsNames<HTriangleMeshGroup>(triNames, grp);
    
	std::vector<std::string>::const_iterator it = triNames.begin();
	for(;it!=triNames.end();++it)
		addTriangleMesh(*it, new ATriangleMeshGroup);

	return true;
}

bool HesperisFile::readTriangle()
{
	bool allValid = true;
	std::map<std::string, ATriangleMeshGroup *>::iterator it = m_triangleMeshes.begin();
	for(; it != m_triangleMeshes.end(); ++it) {
		std::cout<<" read triangle mesh "<<it->first<<"\n";
		HTriangleMeshGroup grp(it->first);
		if(!grp.load(it->second)) {
			std::cout<<" cannot load "<<it->first;
			allValid = false;
		}
		grp.close();
	}
	
	if(!allValid)
		std::cout<<" encounter problem(s) reading triangles.\n";

	return allValid;
}

void HesperisFile::extractCurves(GeometryArray * dst)
{
	dst->create(m_curves.size());
	unsigned i = 0;
	std::map<std::string, CurveGroup *>::const_iterator it = m_curves.begin();
	for(; it != m_curves.end(); ++it) {
		dst->setGeometry(it->second, i);
		i++;
	}
	dst->setNumGeometries(i);
}

void HesperisFile::extractTetrahedronMeshes(GeometryArray * dst)
{
    dst->create(m_terahedrons.size());
	unsigned i = 0;
	std::map<std::string, ATetrahedronMeshGroup *>::const_iterator it = m_terahedrons.begin();
	for(; it != m_terahedrons.end(); ++it) {
		dst->setGeometry(it->second, i);
		i++;
	}
	dst->setNumGeometries(i);
}

void HesperisFile::extractTriangleMeshes(GeometryArray * dst)
{
	dst->create(m_triangleMeshes.size());
	unsigned i = 0;
	std::map<std::string, ATriangleMeshGroup *>::const_iterator it = m_triangleMeshes.begin();
	for(; it != m_triangleMeshes.end(); ++it) {
		dst->setGeometry(it->second, i);
		i++;
	}
	dst->setNumGeometries(i);
}

std::string HesperisFile::WorldPath(const std::string & name)
{ 
	if(SHelper::IsPullPath(name))
		return boost::str(boost::format("/world%1%") % name); 
	return boost::str(boost::format("/world/%1%") % name); 
}

std::string HesperisFile::LocalPath(const std::string & name)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(str, sep);
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter) {
	    if(*tok_iter != "world")
		    r = r + "|" +(*tok_iter);
	}
		
	return r;
}

std::string HesperisFile::checkPath(const std::string & name) const
{
	std::string r(name);
	if(DoStripNamespace) SHelper::removeAnyNamespace(r);
	
	return HObject::ValidPathName(r);
}

bool HesperisFile::writeFrames()
{
	if(!Frames.isValid()) return false;
	HFrameRange g("/world/.frames");
	g.save(&Frames);
	g.close();
	return true;
}

bool HesperisFile::readFrames(HBase * grp)
{
	Frames.reset();
	if(!grp->hasNamedChild(".frames")) {
		return false;
	}
	HFrameRange g("/world/.frames");
	g.load(&Frames);
	g.close();
	return true;
}

void HesperisFile::clearTransforms()
{ 
    std::map<std::string, BaseTransform *>::iterator ittrans = m_transforms.begin();
	for(; ittrans != m_transforms.end(); ++ittrans) 
        delete ittrans->second;
    m_transforms.clear(); 
}

void HesperisFile::clearCurves()
{
    std::map<std::string, CurveGroup *>::iterator it = m_curves.begin();
	for(; it != m_curves.end(); ++it) 
        delete it->second;
    m_curves.clear(); 
}

void HesperisFile::clearTriangleMeshes()
{
    std::map<std::string, ATriangleMeshGroup *>::iterator it = m_triangleMeshes.begin();
	for(; it != m_triangleMeshes.end(); ++it) 
        delete it->second;
    m_triangleMeshes.clear(); 
}

void HesperisFile::clearPolygonalMeshes()
{
    std::map<std::string, APolygonalMesh *>::iterator it = m_polyMeshes.begin();
	for(; it != m_polyMeshes.end(); ++it) 
        delete it->second;
    m_polyMeshes.clear(); 
}

void HesperisFile::clearTetrahedronMeshes()
{
    std::map<std::string, ATetrahedronMeshGroup *>::iterator it = m_terahedrons.begin();
	for(; it != m_terahedrons.end(); ++it) 
        delete it->second;
    m_terahedrons.clear(); 
}

void HesperisFile::clearAttributes()
{
	std::map<std::string, AAttribute *>::iterator it = m_attribs.begin();
	for(; it != m_attribs.end(); ++it) 
        delete it->second;
    m_attribs.clear(); 
}

std::string HesperisFile::modifiedTime()
{
	HWorld grpWorld;
	grpWorld.load();
	grpWorld.close();
	return grpWorld.modifiedTimeStr();
}

bool HesperisFile::writeBundle()
{
    std::cout<<"\n write numeric bundle "<<m_bundlePath;
    HNumericBundle grp(WorldPath(m_bundlePath));
		grp.save(m_bundle);
		grp.close();
    return true;
}

void HesperisFile::setBundleEntry(const ABundleAttribute * d,
	                    const std::string & parentName)
{
    m_bundle = d;
    m_bundlePath = boost::str(boost::format("%1%|%2%") % parentName % d->shortName() );
}

}
//:~
