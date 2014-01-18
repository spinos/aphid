/*
 *  MlScene.h
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HFile.h>
#include <MlFeatherCollection.h>
#include <LightGroup.h>
#include <RenderOptions.h>
class AccPatchMesh;
class MlSkin;
class MlFeather;
class HBase;
class BakeDeformer;
class PlaybackControl;

class MlScene : public HFile, public MlFeatherCollection, public LightGroup, public RenderOptions {
public:
	MlScene();
	virtual ~MlScene();
	
	virtual void setFeatherTexture(const std::string & name);
	std::string featherEditBackground() const;
	
	void setFeatherDistributionMap(const std::string & name);
	std::string featherDistributionMap() const;

	MlSkin * skin();
	AccPatchMesh * body();
	BakeDeformer * bodyDeformer();
	PlaybackControl * playback();
	void setPlayback(PlaybackControl * p);
	
	virtual bool shouldSave();
	virtual void doClear();
	virtual void doClose();
	virtual bool doWrite(const std::string & fileName);
	virtual bool doRead(const std::string & fileName);
	
	bool readBakeFromFile(const std::string & fileName);
	char deformBody(int x);
	
	void enableDeformer();
	void disableDeformer();
	
	void delayLoadBake();
	void bakeRange(int & low, int & high) const;
	
	virtual void setMaxSubdiv(int x);
protected:
	void prepareRender();
	virtual void importBody(const std::string & fileName);
	virtual void afterOpen();
	std::string validateFileExtension(const std::string & fileName) const;
private:
	void writeFeatherExamples();
	void readFeatherExamples();
	void writeFeatherEidtBackground(HBase * g);
	void readFeatherEidtBackground(HBase * g);
	void writeFeatherDistribution(HBase * g);
	void readFeatherDistribution(HBase * g);
	void writeSmoothWeight(HBase * g);
	void readSmoothWeight(HBase * g);
private:
	MlSkin * m_skin;
	AccPatchMesh * m_accmesh;
	BakeDeformer * m_deformer;
	PlaybackControl * m_playback;
	std::string m_featherEditBackgroundName;
	std::string m_featherDistributionName;
	std::string m_bakeName;
};