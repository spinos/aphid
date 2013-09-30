/*
 *  MlScene.h
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseScene.h>
#include <MlFeatherCollection.h>
class AccPatchMesh;
class MlSkin;
class MlFeather;
class MlScene : public BaseScene, public MlFeatherCollection {
public:
	MlScene();
	virtual ~MlScene();

	MlSkin * skin();
	AccPatchMesh * body();
	
	bool shouldSave();
	virtual void clearScene();
	virtual bool writeSceneToFile(const std::string & fileName);
	virtual bool readSceneFromFile(const std::string & fileName);
	
private:
	void writeFeatherExamples();
	void readFeatherExamples();
	
private:
	MlSkin * m_skin;
	AccPatchMesh * m_accmesh;
};