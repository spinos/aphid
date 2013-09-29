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
#include <sstream>
MlScene::MlScene() 
{
	m_accmesh = new AccPatchMesh;
	m_skin = new MlSkin;
	initializeFeatherExample();
}

MlScene::~MlScene() {}

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
	std::vector<MlFeather *>::iterator it;
	for(it = m_feathers.begin(); it != m_feathers.end(); ++it) {
		delete *it;
	}
	m_feathers.clear();
	m_skin->cleanup();
	m_accmesh->cleanup();
	BaseScene::clearScene();
}

bool MlScene::shouldSave()
{
	if(m_accmesh->getNumVertices() < 1)
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
	
	//HMesh grpSkin("/world/skin");
	//grpSkin.close();
	
	grpWorld.close();
	/*	
	HIntAttribute rootAttr("/.range");
	rootAttr.create(4);
	rootAttr.open();
	
	int vrange[4];
	vrange[0] = 31;
	vrange[1] = 1037;
	vrange[2] = -87;
	vrange[3] = 7;
	if(!rootAttr.write(vrange)) std::cout<<"/.range write failed\n";
	rootAttr.close();
	
	HFloatAttribute fltAttr("/.time");
	fltAttr.create(2);
	fltAttr.open();
	
	float vtime[2];
	vtime[0] = .00947;
	vtime[1] = -36.450;
	if(!fltAttr.write(vtime)) std::cout<<"/.time write failed\n";
	fltAttr.close();
	
	HGroup grpAC("/A1/C");
	grpAC.create();
	
	HGroup grpB("/B2");
	grpB.create();
	
	HGroup grpBD("/B2/D");
	grpBD.create();
	
	HGroup grpBDE("/B2/D/E");
	grpBDE.create();
	
	HDataset dsetAg("/A1/g");
	dsetAg.create(32,1);
	dsetAg.open();
	dsetAg.write();
	dsetAg.close();
	
	HDataset dsetBg("/B2/D/g");
	dsetBg.create(32,1);
	dsetBg.open();
	dsetBg.write();
	dsetBg.close();
	*/
	HObject::FileIO.close();

	std::cout<<" Scene file "<<fileName<<" saved at "<<grpWorld.modifiedTimeStr()<<"\n";
	return true;
}

void MlScene::writeFeatherExamples()
{
	HBase g("/world/feathers");
	for(short i = 0; i < numFeatherExamples(); i++) {
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
	sortFeatherExamples();
	initializeFeatherExample();
	//HMesh grpSkin("/world/skin");
	//grpSkin.close();
	
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
		MlFeather * f = addFeatherExample();
		std::stringstream sst;
		sst.str("");
		sst<<"/world/feathers/feather_"<<i;
		HFeather h(sst.str());
		h.load(f);
		h.close();
	}
	g.close();	
}

MlFeather * MlScene::addFeatherExample()
{
	MlFeather * f = new MlFeather;
	f->setFeatherId(numFeatherExamples());
	m_feathers.push_back(f);
	return f;
}

short MlScene::numFeatherExamples() const
{
	return (short)m_feathers.size();
}

void MlScene::selectFeatherExample(short x)
{
	m_selectedFeatherId = x;
}

MlFeather * MlScene::selectedFeatherExample() const
{
	return m_feathers[m_selectedFeatherId];
}

MlFeather * MlScene::featherExample(short idx) const
{
	return m_feathers[idx];
}

void MlScene::initializeFeatherExample()
{
	if(numFeatherExamples() < 1) {
		MlFeather * f = addFeatherExample();
		f->defaultCreate();
		selectFeatherExample(0);
	}
}

void MlScene::sortFeatherExamples()
{
	if(numFeatherExamples() < 2) return;
	int i, j;
	char swapOccured;
	for(i = 0; i < numFeatherExamples() && swapOccured; i++) {
		swapOccured = 0;
		for(j = 0; j < numFeatherExamples() - 1; j++) {
			if(m_feathers[j]->featherId() > m_feathers[j + 1]->featherId()) {
				MlFeather *f = m_feathers[j];
				m_feathers[j] = m_feathers[j + 1];
				m_feathers[j + 1] = f;
				swapOccured = 1;
			}
		}
	}
}
