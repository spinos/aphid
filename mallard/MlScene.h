/*
 *  MlScene.h
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
#include <BaseScene.h>
class AccPatchMesh;
class MlSkin;
class MlFeather;
class MlScene : public BaseScene {
public:
	MlScene();
	virtual ~MlScene();

	MlSkin * skin();
	AccPatchMesh * body();
	
	MlFeather * addFeatherExample();
	short numFeatherExamples() const;
	void selectFeatherExample(short x);
	MlFeather * selectedFeatherExample() const;
	MlFeather * featherExample(short idx) const;
	
	bool shouldSave();
	virtual void clearScene();
	virtual bool writeSceneToFile(const std::string & fileName);
	virtual bool readSceneFromFile(const std::string & fileName);
	
	void initializeFeatherExample();
	
private:
	void writeFeatherExamples();
	void readFeatherExamples();
	void sortFeatherExamples();
private:
	std::vector<MlFeather *> m_feathers;
	MlSkin * m_skin;
	AccPatchMesh * m_accmesh;
	short m_selectedFeatherId;
};