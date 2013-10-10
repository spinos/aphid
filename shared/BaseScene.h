/*
 *  BaseScene.h
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseFile.h>

class BaseScene : public BaseFile {
public:
	BaseScene();
	virtual ~BaseScene();
	
	void setDirty();
	void setClean();
	bool isDirty() const;
	
	bool newScene();
	bool openScene(const std::string & fileName);
	bool saveScene();
	bool saveSceneAs(const std::string & fileName);
	bool revertScene();
	
	virtual bool shouldSave();
	virtual bool discardConfirm();
	virtual void clearScene();
	virtual bool writeSceneToFile(const std::string & fileName);
	virtual bool readSceneFromFile(const std::string & fileName);
	
private:
	bool m_isDirty;
};