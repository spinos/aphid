/*
 *  ScenePort.h
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <ManipulateView.h>
#include "MlScene.h"
class ScenePort : public ManipulateView, public MlScene {
	Q_OBJECT
public:
	ScenePort(QWidget *parent = 0);
	virtual ~ScenePort();

	void finishCreateFeather();
	void finishEraseFeather();
	void deselectFeather();
	
	virtual void setUseDisplaySize(bool x);
	virtual bool selectFeatherExample(unsigned x);
	virtual void beforeSave();
	QString openSheet(QString name);
	
	virtual bool confirmDiscardChanges();
	virtual std::string chooseOpenFileName();
	virtual std::string chooseSaveFileName();
	
	virtual void setFeatherTexture(const std::string & name);

	int m_featherTexId;
	
signals:
	void featherSelectionChanged();
	void sendMessage(QString msg);
	void sceneNameChanged(QString name);
	void sceneOpened();
public slots:
	void receiveCameraChanged();
	void receiveBarbChanged();
	void receivePaintMode(int x);
	void importBodyMesh();
	void importFeatherDistributionMap();
	void chooseBake();
	void receiveFeatherEditBackground(QString name);
	void receiveFeatherAdded();
	void cleanSheet();
	void saveSheet();
	void saveSheetAs();
    void revertSheet();
	void tagFace(bool x);
	void setDisplayFeather(bool x);
	void selectMap(int i);
	
protected:
	virtual PatchMesh * activeMesh() const;
	virtual void resizeEvent( QResizeEvent * event );
	virtual void focusOutEvent(QFocusEvent * event);
	virtual void clientDeselect(QMouseEvent *event);
	virtual void clearSelection();
	
	void showLights() const;
	virtual char selectLight(const Ray & incident);
	
	void loadFeatherDistribution(const std::string & name);
	virtual void importBody(const std::string & fileName);
	
	void selectRegion();
	virtual char selectFeather();
	virtual char floodFeather();
	void paintTexture();
	
	void testPatch();
	void testCurvature();
	
	bool shouldDisplayFeather() const;
private:
	bool m_displayFeather;
};