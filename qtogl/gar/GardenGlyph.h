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

class GlyphPort;

class GardenGlyph : public QGraphicsPathItem
{
	int m_glyphType;
	
public:
	enum { Type = UserType + 1 };
	
	GardenGlyph(const QPixmap & iconPix,
			QGraphicsItem * parent = 0 );
	
	GlyphPort * addPort(const QString & name, 
							bool isOutgoing);
							
	void finalizeShape();
	void moveBlockBy(const QPointF & dp);
	
	int type() const { return Type; }
	
	void setGlyphType(int x);
	const int & glyphType() const;
	
protected:
	void resizeBlock(int bx, int by);
	
private:
	void centerIcon();
	void movePorts(int n, bool downSide);
	
private:
	QGraphicsPixmapItem * m_icon;
	int m_blockWidth, m_blockHeight;
	
};

#endif