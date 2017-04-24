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

class GardenGlyph;
class GlyphPort;
class PlantPiece;
class VegetationPatch;

class ShrubScene : public QGraphicsScene
{
    Q_OBJECT

public:
	ShrubScene(QObject *parent = 0);
	
	void genPlants(VegetationPatch * vege);
	
protected:
/// from ground up to leaf
	void assemblePlant(PlantPiece * pl, GardenGlyph * gl);
	void addBranch(PlantPiece * pl, const GlyphPort * pt);
	void addGrassBranch(PlantPiece * pl, GardenGlyph * gl);
	
private:
};
#endif