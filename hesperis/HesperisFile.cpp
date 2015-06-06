/*
 *  HesperisFile.cpp
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisFile.h"
#include <AllHdf.h>
#include <HWorld.h>
#include <HCurveGroup.h>
#include <BaseBuffer.h>
#include <HTetrahedronMesh.h>
#include <HTriangleMesh.h>
#include <ATriangleMesh.h>
#include <sstream>
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
		default:
			break;
	}
	
	grpWorld.close();
	HObject::FileIO.close();
	
	std::cout<<" finished writing hesperis file at "<<grpWorld.modifiedTimeStr()<<"\n";
	
	return true;
}

bool HesperisFile::writeCurve()
{
	std::stringstream sst;
	std::map<std::string, CurveGroup *>::iterator itcurve = m_curves.begin();
	for(; itcurve != m_curves.end(); ++itcurve) {
		sst.str("");
		sst<<"/world/"<<itcurve->first;
		std::cout<<" write curve "<<sst.str()<<"\n";
		HCurveGroup grpCurve(sst.str());
		grpCurve.save(itcurve->second);
		grpCurve.close();
	}
	return true;
}

bool HesperisFile::writeTetrahedron()
{
	std::stringstream sst;
	std::map<std::string, ATetrahedronMesh *>::iterator it = m_terahedrons.begin();
	for(; it != m_terahedrons.end(); ++it) {
		sst.str("");
		sst<<"/world/"<<it->first;
		std::cout<<" write tetrahedron mesh "<<sst.str()<<"\n";
		HTetrahedronMesh grp(sst.str());
		grp.save(it->second);
		grp.close();
	}
	return true;
}

bool HesperisFile::writeTriangle()
{
	std::stringstream sst;
	std::map<std::string, ATriangleMesh *>::iterator it = m_triangleMeshes.begin();
	for(; it != m_triangleMeshes.end(); ++it) {
		sst.str("");
		sst<<"/world/"<<it->first;
		std::cout<<" write triangle mesh "<<sst.str()<<"\n";
		HTriangleMesh grp(sst.str());
		grp.save(it->second);
		grp.close();
	}
	return true;
}

void HesperisFile::addCurve(const std::string & name, CurveGroup * data)
{ m_curves[name] = data; }

void HesperisFile::addTetrahedron(const std::string & name, ATetrahedronMesh * data)
{ m_terahedrons[name] = data; }

void HesperisFile::addTriangleMesh(const std::string & name, ATriangleMesh * data)
{ m_triangleMeshes[name] = data; }

bool HesperisFile::doRead(const std::string & fileName)
{
	if(!HFile::doRead(fileName)) return false;
	
	std::cout<<" reading curves file "<<fileName<<"\n";
	HWorld grpWorld;
	grpWorld.load();
	
	switch (m_readComp) {
		case RCurve:
			readCurve();
			break;
		case RTetra:
			readTetrahedron();
			break;
		case RTri:
			listTriangle(&grpWorld);
			readTriangle();
		default:
			break;
	}
	
	grpWorld.close();
	
	std::cout<<" finished reading hesperis file modified at "<<grpWorld.modifiedTimeStr()<<"\n";
	
	return true;
}

bool HesperisFile::readCurve()
{
	bool allValid = true;
	std::stringstream sst;
	std::map<std::string, CurveGroup *>::iterator itcurve = m_curves.begin();
	for(; itcurve != m_curves.end(); ++itcurve) {
		sst.str("");
		sst<<"/world/"<<itcurve->first;
		std::cout<<" read curve "<<sst.str()<<"\n";
		HCurveGroup grpCurve(sst.str());
		if(!grpCurve.load(itcurve->second)) {
			std::cout<<" cannot load "<<sst.str();
			allValid = false;
		}
		
		grpCurve.close();
	}
	
	if(!allValid)
		std::cout<<" encounter problem(s) reading curves.\n";

	return allValid;
}

bool HesperisFile::readTetrahedron()
{
    bool allValid = true;
	std::stringstream sst;
	std::map<std::string, ATetrahedronMesh *>::iterator it = m_terahedrons.begin();
	for(; it != m_terahedrons.end(); ++it) {
		sst.str("");
		sst<<"/world/"<<it->first;
		std::cout<<" read tetrahedron mesh "<<sst.str()<<"\n";
		HTetrahedronMesh grp(sst.str());
		if(!grp.load(it->second)) {
			std::cout<<" cannot load "<<sst.str();
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
	std::vector<std::string > triNames;
	grp->lsTypedChild<HTriangleMesh>(triNames);
	
	std::vector<std::string>::const_iterator it = triNames.begin();
	for(;it!=triNames.end();++it) {
		addTriangleMesh(*it, new ATriangleMesh);
	}
	return true;
}

bool HesperisFile::readTriangle()
{
	bool allValid = true;
	std::map<std::string, ATriangleMesh *>::iterator it = m_triangleMeshes.begin();
	for(; it != m_triangleMeshes.end(); ++it) {
		std::cout<<" read triangle mesh "<<it->first<<"\n";
		HTriangleMesh grp(it->first);
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
//:~