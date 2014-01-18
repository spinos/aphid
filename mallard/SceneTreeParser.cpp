/*
 *  SceneTreeParser.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "SceneTreeParser.h"
#include "SceneTreeItem.h"
#include "MlScene.h"
#include "AllLight.h"
#include "AllEdit.h"

SceneTreeParser::SceneTreeParser(const QStringList &headers, MlScene* scene, 
                     QObject *parent) : SceneTreeModel(headers, parent)
{
	m_scene = scene;
    setupModelData(getRootItem());
}

SceneTreeParser::~SceneTreeParser() {}

void SceneTreeParser::rebuild()
{
	clear();
	setupModelData(getRootItem());
}

void SceneTreeParser::setupModelData(SceneTreeItem *parent)
{
    QList<SceneTreeItem*> parents;
    parents << parent;
	addOptions(parents);
	addLights(parents);
}

void SceneTreeParser::addOptions(QList<SceneTreeItem*> & parents)
{
	addBase(parents, "options", 0);
	addIntAttr(parents, "max_subdiv", 1, m_scene->maxSubdiv());
	addIntAttr(parents, "AA_samples", 1, m_scene->AASample());
	addIntAttr(parents, "res_x", 1, m_scene->renderImageWidth());
	addIntAttr(parents, "res_y", 1, m_scene->renderImageHeight());
	addBolAttr(parents, "use_display_size", 1, m_scene->useDisplaySize());
}

void SceneTreeParser::addLights(QList<SceneTreeItem*> & parents)
{
	addBase(parents, "lights", 0);
	unsigned nl = m_scene->numLights();
	for(unsigned i = 0; i < nl; i++) {
		BaseLight * l = m_scene->getLight(i);
		addBase(parents, l->name(), 1);
		addFltAttr(parents, "intensity", 2, l->intensity());
		addIntAttr(parents, "samples", 2, l->samples());
		addBolAttr(parents, "cast_shadow", 2, l->castShadow());
		Float3 fc = l->lightColor();
		QColor col;
		col.setRgbF(fc.x, fc.y, fc.z);
		addRGBAttr(parents, "light_color", 2, col);
	}
}

void SceneTreeParser::receiveData(QWidget * editor)
{
	QModelEdit * me = static_cast<QModelEdit *>(editor);
	SceneTreeItem *item = getItem(me->index());
	updateScene(item);
}

void SceneTreeParser::updateScene(SceneTreeItem * item)
{
	const QString baseName = item->fullPathName().last();
	if(baseName == "options") {
		updateOptions(item);
	}
	else if(baseName == "lights") {
		updateLights(item);
	}
}

void SceneTreeParser::updateOptions(SceneTreeItem * item)
{
	const QString attrName = item->fullPathName().first();
	if(attrName == "max_subdiv") {
		qDebug()<<"set maxsubdiv "<<item->data(1).toInt();
		m_scene->setMaxSubdiv(item->data(1).toInt());
	}
	else if(attrName == "AA_samples") {
		qDebug()<<"set aa "<<item->data(1).toInt();
		m_scene->setAASample(item->data(1).toInt());
	}
	else if(attrName == "res_x") {
		qDebug()<<"set resx "<<item->data(1).toInt();
		m_scene->setRenderImageWidth(item->data(1).toInt());
	}
	else if(attrName == "res_y") {
		qDebug()<<"set resy "<<item->data(1).toInt();
		m_scene->setRenderImageHeight(item->data(1).toInt());
	}
	else if(attrName == "use_display_size") {
		qDebug()<<"set use display size "<<item->data(1).toBool();
		m_scene->setUseDisplaySize(item->data(1).toBool());
	}
}

void SceneTreeParser::updateLights(SceneTreeItem * item)
{
	const QString lightName = item->fullPathName()[1];
	BaseLight * l = m_scene->getLight(lightName.toStdString());
	if(!l) {
		qDebug()<<"WARNING: cannot find light named "<<lightName;
		return;
	}
	const QString attrName = item->fullPathName().first();
	if(attrName == "intensity") {
		qDebug()<<"set light intensity "<<item->data(1).toDouble();
		l->setIntensity(item->data(1).toFloat());
	}
	else if(attrName == "samples") {
		qDebug()<<"set light smaples "<<item->data(1).toInt();
		l->setSamples(item->data(1).toInt());
	}
	else if(attrName == "cast_shadow") {
		qDebug()<<"set light case shadow "<<item->data(1).toBool();
		l->setCastShadow(item->data(1).toBool());
	}
	else if(attrName == "light_color") {
		QColor c = item->data(1).value<QColor>();
		qDebug()<<"set light color "<<c;
		float r = c.redF();
		float g = c.greenF();
		float b = c.blueF();
		l->setLightColor(r, g, b);
	}
}
