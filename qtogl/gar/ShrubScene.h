/*
 *  ShrubScene.h
 *  garden
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SHRUB_SCENE_H
#define GAR_SHRUB_SCENE_H

#include <QGraphicsScene>

QT_BEGIN_NAMESPACE
class QGraphicsSceneMouseEvent;
class QParallelAnimationGroup;
QT_END_NAMESPACE

class ShrubScene : public QGraphicsScene
{
    Q_OBJECT

public:
	ShrubScene(QObject *parent = 0);
	
protected:
	
};
#endif