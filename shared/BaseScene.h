/*
 *  BaseScene.h
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>

class BaseScene {
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
	
	void setLatestError(BaseScene::ErrorMsg err);
	BaseScene::ErrorMsg latestError() const;
	
	virtual bool discardConfirm();
	virtual void clearScene();
	virtual bool writeSceneToFile(const std::string & fileName);
	virtual bool readSceneFromFile(const std::string & fileName);
	
	static bool fileExists(const std::string & fileName);
	
	std::string fileName() const;
private:
	std::string m_fileName;
	ErrorMsg m_error;
	bool m_isDirty;
};