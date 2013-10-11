/*
 *  MlScene.h
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseFile.h>
#include <MlFeatherCollection.h>
class AccPatchMesh;
class MlSkin;
class MlFeather;
class HBase;
class BakeDeformer;
class PlaybackControl;

class MlScene : public BaseFile, public MlFeatherCollection {
public:
	MlScene();
	virtual ~MlScene();
	
	void setFeatherEditBackground(const std::string & name);
	std::string featherEditBackground() const;

	MlSkin * skin();
	AccPatchMesh * body();
	BakeDeformer * bodyDeformer();
	PlaybackControl * playback();
	void setPlayback(PlaybackControl * p);
	
	virtual bool shouldSave();
	virtual void doClear();
	virtual bool doWrite(const std::string & fileName);
	virtual bool doRead(const std::string & fileName);
	
	bool readBakeFromFile(const std::string & fileName);
	char deformBody(int x);
	
	void enableDeformer();
	void disableDeformer();
	
	void delayLoadBake();
private:
	void writeFeatherExamples();
	void readFeatherExamples();
	void writeFeatherEidtBackground(HBase * g);
	void readFeatherEidtBackground(HBase * g);
private:
	MlSkin * m_skin;
	AccPatchMesh * m_accmesh;
	BakeDeformer * m_deformer;
	PlaybackControl * m_playback;
	std::string m_featherEditBackgroundName;
	std::string m_bakeName;
};