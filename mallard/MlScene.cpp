/*
 *  MlScene.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlScene.h"
#include <AccPatchMesh.h>
#include <MlSkin.h>
#include <MlFeather.h>
#include <AllHdf.h>
#include <HWorld.h>
#include <HMesh.h>
#include <HFeather.h>
#include <HSkin.h>
#include <HLight.h>
#include <HOption.h>
#include <BakeDeformer.h>
#include <PlaybackControl.h>
#include <sstream>
#include <EasemodelUtil.h>
#include <boost/filesystem.hpp>
void test()
{
	std::cout<<"\nh io begin\n";
	HObject::FileIO.open("parttest.f", HDocument::oCreate);
	
	int n = 33;
	float data[10], rd[33];
	for(int i=0; i < 10; i++) data[i] = i + 2;
	
	HBase grp("/world");
	if(!grp.hasNamedData(".t"))
		grp.addFloatData(".t", n);
		
	HDataset::SelectPart p;
	p.start[0] = 0;
	p.count[0] = 1;
	p.block[0] = 10; 
		
	grp.writeFloatData(".t", n, data, &p);

	for(int i=0; i < 10; i++) data[i] = i + 10;
	
	p.start[0] = 19;
	grp.writeFloatData(".t", n, data, &p);
	
	grp.close();
	HObject::FileIO.close();
	
	HObject::FileIO.open("parttest.f", HDocument::oReadAndWrite);
	HBase grpi("/world");
	grpi.readFloatData(".t", n, rd);
	std::cout<<"\n";
	for(int i=0; i < n; i++) std::cout<<" "<<rd[i];
	
	float rd9[10];
	grpi.readFloatData(".t", n, rd9, &p);
	std::cout<<"\n";
	for(int i=0; i < 10; i++) std::cout<<" "<<rd9[i];
	
	grpi.close();
	HObject::FileIO.close();
	
	std::cout<<"\nh io end\n";
}

MlScene::MlScene() 
{
	m_accmesh = new AccPatchMesh;
	m_skin = new MlSkin;
	m_deformer = new BakeDeformer;
	m_featherEditBackgroundName = "unknown";
	m_playback = 0;
}

MlScene::~MlScene() 
{
	clearFeatherExamples();
	delete m_skin;
	delete m_accmesh;
	delete m_deformer;
}

void MlScene::setFeatherTexture(const std::string & name)
{
	m_featherEditBackgroundName = name;
}

std::string MlScene::featherEditBackground() const
{
	return m_featherEditBackgroundName;
}

void MlScene::setFeatherDistributionMap(const std::string & name)
{
	m_featherDistributionName = name;
}

std::string MlScene::featherDistributionMap() const
{
	return m_featherDistributionName;
}

MlSkin * MlScene::skin()
{
	return m_skin;
}

AccPatchMesh * MlScene::body()
{
	return m_accmesh;
}

BakeDeformer * MlScene::bodyDeformer()
{
	return m_deformer;
}

PlaybackControl * MlScene::playback()
{
	return m_playback;
}

void MlScene::setPlayback(PlaybackControl * p)
{
	m_playback = p;
}

bool MlScene::shouldSave()
{
	if(!HFile::shouldSave()) return false;
	if(m_accmesh->isEmpty()) return false;
	if(m_skin->numFeathers() < 1) return false;
	return true;
}

void MlScene::doClear()
{
	clearFeatherExamples();
	initializeFeatherExample();
	disableDeformer();
	delete m_skin;
	m_skin = new MlSkin;
	delete m_accmesh;
	m_accmesh = new AccPatchMesh;
	delete m_deformer;
	m_deformer = new BakeDeformer;
	
	m_featherEditBackgroundName = "unknown";
	m_featherDistributionName = "unknown";
	
	HFile::doClear();
	clearLights();
}

void MlScene::doClose() 
{
	HFile::doClose();
}

bool MlScene::doWrite(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseFile::FileNotWritable);
		return false;
	}
	
	std::cout<<"write scene to "<<fileName<<"\n";
	
	HWorld grpWorld;
	grpWorld.save();
	
	HOption grpOpt("/world/options");
	grpOpt.save(this);
	
	HMesh grpBody("/world/body");
	grpBody.save(m_accmesh);
	
	if(grpBody.hasNamedAttr(".bakefile"))
		grpBody.discardNamedAttr(".bakefile");
		
	if(m_deformer->isEnabled()) {
		grpBody.addStringAttr(".bakefile", m_deformer->fileName().size());
		grpBody.writeStringAttr(".bakefile", m_deformer->fileName());
	}
	
	writeFeatherDistribution(&grpBody);
	writeSmoothWeight(&grpBody);
	
	grpBody.close();
	
	writeFeatherExamples();
	
	HSkin grpSkin("/world/skin");
	grpSkin.save(m_skin);
	grpSkin.close();
	
	HLight grpLight("/world/lights");
	grpLight.save(this);
	grpLight.close();
	
	grpWorld.close();

	HObject::FileIO.close();

	std::cout<<" Scene file "<<fileName<<" saved at "<<grpWorld.modifiedTimeStr()<<"\n";
	return true;
}

void MlScene::writeFeatherExamples()
{
	HBase g("/world/feathers");
	
	writeFeatherEidtBackground(&g);
	
	for(MlFeather * f = firstFeatherExample(); hasFeatherExample(); f = nextFeatherExample()) {
		if(!f) continue;
		std::stringstream sst;
		sst.str("");
		sst<<"/world/feathers/feather_"<<f->featherId();
		HFeather h(sst.str());
		h.save(f);
		h.close();
	}
	g.close();
}

void MlScene::writeFeatherEidtBackground(HBase * g)
{
	if(m_featherEditBackgroundName == "unknown") return;
	if(!g->hasNamedAttr(".bkgrd"))
		g->addStringAttr(".bkgrd", m_featherEditBackgroundName.size());
		
	g->writeStringAttr(".bkgrd", m_featherEditBackgroundName);
}

void MlScene::writeFeatherDistribution(HBase * g)
{
	if(m_featherDistributionName == "unknown") return;
	if(!g->hasNamedAttr(".distrmp"))
		g->addStringAttr(".distrmp", m_featherDistributionName.size());
		
	g->writeStringAttr(".distrmp", m_featherDistributionName);
}

void MlScene::writeSmoothWeight(HBase * g)
{
	if(m_accmesh->hasVertexData("weishell")) {
		if(!g->hasNamedData("weishell"))
			g->addFloatData("weishell", m_accmesh->getNumVertices());

		g->writeFloatData("weishell", m_accmesh->getNumVertices(), m_accmesh->perVertexFloat("weishell"));
	}
}

bool MlScene::doRead(const std::string & fileName)
{
    if(!HFile::doRead(fileName)) return false;
	/*if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}*/
	
	std::cout<<"read scene from "<<fileName<<"\n";
	
	HWorld grpWorld;
	grpWorld.load();
	
	HOption grpOpt("/world/options");
	grpOpt.load(this);

	HMesh grpBody("/world/body");
	grpBody.load(m_accmesh);
	
	m_bakeName = std::string("");
	if(grpBody.hasNamedAttr(".bakefile")) {
		grpBody.readStringAttr(".bakefile", m_bakeName);
	}
	
	readFeatherDistribution(&grpBody);
	readSmoothWeight(&grpBody);
	
	grpBody.close();
	
	readFeatherExamples();
	initializeFeatherExample();
	
	HSkin grpSkin("/world/skin");
	grpSkin.load(m_skin);
	grpSkin.close();
	
	HLight grpLight("/world/lights");
	grpLight.load(this);
	grpLight.close();
	
	grpWorld.close();
	//HObject::FileIO.close();
	
	std::cout<<" Scene file "<<fileName<<" modified at "<<grpWorld.modifiedTimeStr()<<"\n";
	
	return true;
}

void MlScene::readFeatherExamples()
{
	HBase g("/world/feathers");
	
	readFeatherEidtBackground(&g);
	
	int nf = g.numChildren();
	for(int i = 0; i < nf; i++) {
		std::stringstream sst;
		sst.str("");
		sst<<"/world/feathers/"<<g.getChildName(i);
		HFeather h(sst.str());
		
		int fid = h.loadId();
		MlFeather * f = featherExample(fid);
		if(!f)
			f = addFeatherExampleId(fid);

		h.load(f);
		h.close();
	}
	g.close();	
}

void MlScene::readFeatherEidtBackground(HBase * g)
{
	if(!g->hasNamedAttr(".bkgrd")) return;
		
	g->readStringAttr(".bkgrd", m_featherEditBackgroundName);
}

void MlScene::readFeatherDistribution(HBase * g)
{
	if(!g->hasNamedAttr(".distrmp")) return;
		
	g->readStringAttr(".distrmp", m_featherDistributionName);
}

void MlScene::readSmoothWeight(HBase * g)
{
	if(!g->hasNamedData("weishell")) return;

	g->readFloatData("weishell", m_accmesh->getNumVertices(), m_accmesh->perVertexFloat("weishell"));
}

bool MlScene::readBakeFromFile(const std::string & fileName)
{
	if(!FileExists(fileName)) {
		setLatestError(BaseFile::FileNotFound);
		return false;
	}
	
	if(!m_deformer->open(fileName)) return false;
		
	m_deformer->verbose();
	m_deformer->setMesh(m_accmesh);
	enableDeformer();
	return true;
}

char MlScene::deformBody(int x)
{
	m_deformer->setCurrentFrame(x);
	if(!m_deformer->solve()) return false;
	m_accmesh->update(m_skin->topology());
	m_skin->computeVertexDisplacement();
	return true;
}

void MlScene::enableDeformer()
{
	m_deformer->enable();
	if(m_playback) {
	    m_playback->setFrameRange(m_deformer->minFrame(), m_deformer->maxFrame());
	    m_playback->enable();
	}
}
	
void MlScene::disableDeformer()
{
	if(m_deformer->isEnabled()) {
		m_deformer->disable();
		m_deformer->close();
	}
	if(m_playback) 
	    m_playback->disable();
}

void MlScene::delayLoadBake()
{
	if(m_bakeName == "") return;
	readBakeFromFile(m_bakeName);
}

void MlScene::bakeRange(int & low, int & high) const
{
    low = m_deformer->minFrame();
    high = m_deformer->maxFrame();
}

void MlScene::prepareRender()
{
	int i = 99;
	for(MlFeather * f = firstFeatherExample(); hasFeatherExample(); f = nextFeatherExample()) {
		if(!f) continue;
		f->setSeed(i++);
		f->computeNoise();
	}
}

void MlScene::importBody(const std::string & fileName)
{
    m_skin->finishCreateFeather();
	m_skin->discardActive();
	disableDeformer();
	delete m_skin;
	m_skin = new MlSkin;
	delete m_accmesh;
	m_accmesh = new AccPatchMesh;
	delete m_deformer;
	m_deformer = new BakeDeformer;
	ESMUtil::ImportPatch(fileName.c_str(), m_accmesh);
	m_skin->setBodyMesh(m_accmesh);
	m_accmesh->setup(m_skin->topology());
	m_deformer->setMesh(m_accmesh);
	m_skin->computeFaceCalamusIndirection();
	m_skin->computeVertexDisplacement();
	setCollision(m_skin);
}

void MlScene::afterOpen()
{
	m_accmesh->putIntoObjectSpace();
	m_skin->setBodyMesh(m_accmesh);
	m_accmesh->setup(m_skin->topology());
	m_deformer->setMesh(m_accmesh);
	delayLoadBake();
	m_accmesh->update(m_skin->topology());
	m_skin->computeFaceCalamusIndirection();
	m_skin->computeVertexDisplacement();
	setCollision(m_skin);
}

std::string MlScene::validateFileExtension(const std::string & fileName) const
{
	boost::filesystem::path p(fileName);
	if(p.extension() != ".mal") p = p.replace_extension(".mal");
	return p.string();
}
