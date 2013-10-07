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
	enum ErrorMsg {
		NoError = 0,
		FileNotFound = 1,
		FileNotWritable = 2
	};
	
	BaseScene();
	virtual ~BaseScene();
	
	void setDirty();
	void setClean();
	bool isDirty() const;
	bool isUntitled() const;
	
	bool newScene();
	bool openScene(const std::string & fileName);
	bool saveScene();
	bool saveSceneAs(const std::string & fileName);
	bool revertScene();
	
	void setLatestError(BaseScene::ErrorMsg err);
	BaseScene::ErrorMsg latestError() const;
	
	virtual bool shouldSave();
	virtual bool discardConfirm();
	virtual void clearScene();
	virtual bool writeSceneToFile(const std::string & fileName);
	virtual bool readSceneFromFile(const std::string & fileName);
	
private:
	ErrorMsg m_error;
	bool m_isDirty;
};