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
#include <BakeDeformer.h>
#include <PlaybackControl.h>
#include <sstream>

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
	m_skin->cleanup();
	m_accmesh->cleanup();
}

void MlScene::setFeatherEditBackground(const std::string & name)
{
	m_featherEditBackgroundName = name;
}

std::string MlScene::featherEditBackground() const
{
	return m_featherEditBackgroundName;
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
	if(!BaseFile::shouldSave()) return false;
	if(m_accmesh->isEmpty()) return false;
	if(m_skin->numFeathers() < 1) return false;
	return true;
}

void MlScene::doClear()
{
	clearFeatherExamples();
	initializeFeatherExample();
	m_skin->cleanup();
	m_accmesh->cleanup();
	disableDeformer();
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
	
	HMesh grpBody("/world/body");
	grpBody.save(m_accmesh);
	
	if(grpBody.hasNamedAttr(".bakefile"))
		grpBody.discardNamedAttr(".bakefile");
		
	if(m_deformer->isEnabled()) {
		grpBody.addStringAttr(".bakefile", m_deformer->fileName().size());
		grpBody.writeStringAttr(".bakefile", m_deformer->fileName());
	}
	
	grpBody.close();
	
	writeFeatherExamples();
	
	HSkin grpSkin("/world/skin");
	grpSkin.save(m_skin);
	grpSkin.close();
	
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

bool MlScene::doRead(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	std::cout<<"read scene from "<<fileName<<"\n";
		
	HWorld grpWorld;
	grpWorld.load();

	HMesh grpBody("/world/body");
	grpBody.load(m_accmesh);
	
	m_bakeName = std::string("");
	if(grpBody.hasNamedAttr(".bakefile")) {
		grpBody.readStringAttr(".bakefile", m_bakeName);
	}
	
	grpBody.close();
	
	readFeatherExamples();
	initializeFeatherExample();
	
	HSkin grpSkin("/world/skin");
	grpSkin.load(m_skin);
	grpSkin.close();
	
	grpWorld.close();
	HObject::FileIO.close();
	
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

bool MlScene::readBakeFromFile(const std::string & fileName)
{
	if(!FileExists(fileName)) {
		setLatestError(BaseFile::FileNotFound);
		return false;
	}
	
	if(!m_deformer->open(fileName)) return false;
		
	m_deformer->verbose();
	
	enableDeformer();
	return false;
}

char MlScene::deformBody(int x)
{
	m_deformer->setCurrentFrame(x);
	if(!m_deformer->solve()) return false;
	
	return true;
}

void MlScene::enableDeformer()
{
	m_deformer->enable();
	m_playback->setFrameRange(m_deformer->minFrame(), m_deformer->maxFrame());
	m_playback->enable();
}
	
void MlScene::disableDeformer()
{
	if(m_deformer->isEnabled()) {
		m_deformer->disable();
		m_deformer->close();
	}
	m_playback->disable();
}

void MlScene::delayLoadBake()
{
	if(m_bakeName != "")
		readBakeFromFile(m_bakeName);
}
