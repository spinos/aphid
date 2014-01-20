/*
 *  SceneTreeParser.h
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "SceneTreeModel.h"
class MlScene;

class SceneTreeParser : public SceneTreeModel {
	Q_OBJECT
public:
	SceneTreeParser(const QStringList &headers, MlScene* scene, QObject *parent = 0);
    ~SceneTreeParser();
	
	void rebuild();
public slots:
	void receiveData(QWidget * editor);
	
protected:
	void addOptions(QList<SceneTreeItem*> & parents);
	void addCamera(QList<SceneTreeItem*> & parents);
	void addLights(QList<SceneTreeItem*> & parents);
	void setupModelData(SceneTreeItem *parent);
    void updateScene(SceneTreeItem * item);
	void updateOptions(SceneTreeItem * item);
	void updateLights(SceneTreeItem * item);
private:
	MlScene * m_scene;
};