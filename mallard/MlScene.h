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
class HBase;
class MlScene : public BaseScene, public MlFeatherCollection {
public:
	MlScene();
	virtual ~MlScene();
	
	void setFeatherEditBackground(const std::string & name);
	std::string featherEditBackground() const;

	MlSkin * skin();
	AccPatchMesh * body();
	
	bool shouldSave();
	virtual void clearScene();
	virtual bool writeSceneToFile(const std::string & fileName);
	virtual bool readSceneFromFile(const std::string & fileName);
	
private:
	void writeFeatherExamples();
	void readFeatherExamples();
	void writeFeatherEidtBackground(HBase * g);
	void readFeatherEidtBackground(HBase * g);
private:
	MlSkin * m_skin;
	AccPatchMesh * m_accmesh;
	std::string m_featherEditBackgroundName;
};