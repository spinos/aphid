/*
 *  GardenGlyph.h
 *
 *
 *  Created by jian zhang on 3/31/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_GARDEN_GLYPH_H
#define GAR_GARDEN_GLYPH_H

#include <QGraphicsPathItem>

QT_BEGIN_NAMESPACE
class QPixmap;
class QGraphicsPixmapItem;
QT_END_NAMESPACE

namespace aphid {

class GlyphPort;
class GlyphHalo;

}

class GardenGlyph : public QGraphicsPathItem
{
	int m_glyphType;
	std::string m_glyphName;
	
public:
	enum { Type = UserType + 1 };
	
	GardenGlyph(const QPixmap & iconPix,
			QGraphicsItem * parent = 0 );
	
	aphid::GlyphPort * addPort(const QString & name, 
							bool isOutgoing);
							
	void finalizeShape();
	void moveBlockBy(const QPointF & dp);
	
	int type() const { return Type; }
	
	void setGlyphType(int x);
	const int & glyphType() const;
	
	void setHalo(aphid::GlyphHalo* hal);
	void showHalo();
	void hideHalo();
	aphid::GlyphHalo* halo();
	
	QPointF localCenter() const;
	const std::string& glyphName() const;
	
protected:
	void resizeBlock(int bx, int by);
	virtual void mousePressEvent ( QGraphicsSceneMouseEvent * event );
	virtual void mouseDoubleClickEvent( QGraphicsSceneMouseEvent * event );
	
private:
	void centerIcon();
	void movePorts(int n, bool downSide);
	
private:
	QGraphicsPixmapItem * m_icon;
	aphid::GlyphHalo* m_halo;
	int m_blockWidth, m_blockHeight;
	
};

#endif