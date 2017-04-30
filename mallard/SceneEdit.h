/*
 *  SceneEdit.h
 *  mallard
 *
 *  Created by jian zhang on 1/13/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
class MlScene;
#include <QDialog>
class SceneTreeParser;
QT_BEGIN_NAMESPACE
class QTreeView;
QT_END_NAMESPACE

class SceneEdit : public QDialog
{
    Q_OBJECT
	
public:
	SceneEdit(MlScene * scene, QWidget *parent = 0);
	virtual ~SceneEdit();
	QObject * model() const;
signals:
	
public slots:
	void reloadScene();
private:
	QTreeView * m_view;
	SceneTreeParser * m_model;
};
