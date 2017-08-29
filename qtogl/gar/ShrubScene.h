/*
 *  ShrubScene.h
 *  
 *	glyph selection
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SHRUB_SCENE_H
#define GAR_SHRUB_SCENE_H

#include <syn/PlantAssemble.h>
#include <QGraphicsScene>

QT_BEGIN_NAMESPACE
class QGraphicsSceneMouseEvent;
QT_END_NAMESPACE

class ShrubScene : public QGraphicsScene, public gar::PlantAssemble
{
    Q_OBJECT

public:
	ShrubScene(Vegetation * vege, QObject *parent = 0);
	virtual ~ShrubScene();
	
	GardenGlyph* lastSelectedGlyph();
	const GardenGlyph* lastSelectedGlyph() const;
	const aphid::ATriangleMesh* lastSelectedGeom() const;
	void selectGlyph(GardenGlyph* gl);
	void deselectGlyph();
	
protected:
/// first (selected) ground piece
	virtual GardenGlyph* getGround();
	
private:

private:
	QList<GardenGlyph *> m_selectedGlyph;
	GardenGlyph* m_lastSelectedGlyph;
	
};

#endif