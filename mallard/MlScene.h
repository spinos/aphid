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
class AccPatchMesh;
class MlSkin;
class MlScene : public BaseScene {
public:
	MlScene();
	virtual ~MlScene();

	MlSkin * m_skin;
	AccPatchMesh * m_accmesh;
	
	virtual void clearScene();
	virtual bool writeSceneToFile(const std::string & fileName);
	virtual bool readSceneFromFile(const std::string & fileName);
private:
	
};