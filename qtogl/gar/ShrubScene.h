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
QT_END_NAMESPACE

class GardenGlyph;
class GlyphPort;
class PlantPiece;
class VegetationPatch;
class Vegetation;

class ShrubScene : public QGraphicsScene
{
    Q_OBJECT

public:
	ShrubScene(Vegetation * vege, QObject *parent = 0);
	virtual ~ShrubScene();
	
	void genSinglePlant();
	void genMultiPlant();
		
protected:
/// from ground up to leaf
	void assemblePlant(PlantPiece * pl, GardenGlyph * gl);
	void addBranch(PlantPiece * pl, const GlyphPort * pt);
	void addGrassBranch(PlantPiece * pl, GardenGlyph * gl);
	
private:
/// first ground piece
	GardenGlyph* getGround();
	void genAPlant(VegetationPatch * vege);
	void growOnGround(VegetationPatch * vege, GardenGlyph * gnd);
/// throw darts repeatedly untill patch is full
	void genPlants(VegetationPatch * vege);
/// point and angle
	void genBushPlants(VegetationPatch * vege, GardenGlyph * gnd);
	
private:
	Vegetation * m_vege;
};
#endif