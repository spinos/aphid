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
#include <sstream>

void test()
{
	std::cout<<"\nh io begin\n";
	HObject::FileIO.open("parttest.f", HDocument::oCreate);
	
	int n = 33;
	float data[10], rd[n];
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
}

MlScene::~MlScene() 
{
	clearScene();
}

MlSkin * MlScene::skin()
{
	return m_skin;
}

AccPatchMesh * MlScene::body()
{
	return m_accmesh;
}

void MlScene::clearScene()
{
	clearFeatherExamples();
	initializeFeatherExample();
	m_skin->cleanup();
	m_accmesh->cleanup();
	BaseScene::clearScene();
}

bool MlScene::shouldSave()
{
	if(m_accmesh->getNumVertices() < 1)
		return false;
	if(m_skin->numFeathers() < 1)
		return false;
	return true;
}

bool MlScene::writeSceneToFile(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseScene::FileNotWritable);
		return false;
	}
	
	std::cout<<"write scene to "<<fileName<<"\n";
	
	HWorld grpWorld;
	grpWorld.save();
	
	HMesh grpBody("/world/body");
	grpBody.save(m_accmesh);
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
	for(unsigned i = 0; i < numFeatherExamples(); i++) {
		MlFeather * f = featherExample(i);
		std::stringstream sst;
		sst.str("");
		sst<<"/world/feathers/feather_"<<f->featherId();
		HFeather h(sst.str());
		h.save(f);
		h.close();
	}
	g.close();
}

bool MlScene::readSceneFromFile(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseScene::FileNotWritable);
		return false;
	}
	
	std::cout<<"read scene from "<<fileName<<"\n";
		
	HWorld grpWorld;
	grpWorld.load();

	HMesh grpBody("/world/body");
	grpBody.load(m_accmesh);
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
	int nf = g.numChildren();
	for(int i = 0; i < nf; i++) {
		MlFeather * f = featherExample(i);
		if(!f)
			f = addFeatherExample();
		std::stringstream sst;
		sst.str("");
		sst<<"/world/feathers/feather_"<<i;
		HFeather h(sst.str());
		h.load(f);
		h.close();
	}
	g.close();	
}
