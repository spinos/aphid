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
class BakeDeformer;

class MlScene : public BaseScene, public MlFeatherCollection {
public:
	MlScene();
	virtual ~MlScene();
	
	void setFeatherEditBackground(const std::string & name);
	std::string featherEditBackground() const;

	MlSkin * skin();
	AccPatchMesh * body();
	BakeDeformer * bodyDeformer();
	
	bool shouldSave();
	virtual void clearScene();
	bool writeSceneToFile(const std::string & fileName);
	bool readSceneFromFile(const std::string & fileName);
	bool readBakeFromFile(const std::string & fileName);
	
private:
	void writeFeatherExamples();
	void readFeatherExamples();
	void writeFeatherEidtBackground(HBase * g);
	void readFeatherEidtBackground(HBase * g);
private:
	MlSkin * m_skin;
	AccPatchMesh * m_accmesh;
	BakeDeformer * m_deformer;
	std::string m_featherEditBackgroundName;
};