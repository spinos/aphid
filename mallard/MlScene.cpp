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
#include <AllHdf.h>
#include <HWorld.h>
#include <HMesh.h>
MlScene::MlScene() 
{
	m_accmesh = new AccPatchMesh;
	m_skin = new MlSkin;
}

MlScene::~MlScene() {}

void MlScene::clearScene()
{
	m_skin->cleanup();
	m_accmesh->cleanup();
	BaseScene::clearScene();
}

bool MlScene::writeSceneToFile(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseScene::FileNotWritable);
		return false;
	}
	
	std::cout<<"write scene to "<<fileName;
	
	HWorld grpWorld;
	grpWorld.save();
	
	//grpWorld.load();
	
	HMesh grpBody("/world/body");
	grpBody.save(m_accmesh);
	
	grpBody.close();
	
	//HMesh grpSkin("/world/skin");
	//grpSkin.close();
	
	//if(grpWorld.hasNamedChild("skin")) std::cout<<"found skin";
	
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
/*
	HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite);
	dsetAg.open();
	//dsetAg.read();
	dsetAg.close();
	
	dsetBg.open();
	//dsetBg.read();
	dsetBg.close();
	
	//HObject::FileIO.deleteObject("/A1");
	HObject::FileIO.close();*/
	return true;
}