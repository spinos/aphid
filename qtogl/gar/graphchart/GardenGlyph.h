/*
 *  GardenGlyph.h
 *
 *  top level item holds instance of attribs
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

class PieceAttrib;

class GardenGlyph : public QGraphicsPathItem
{
	PieceAttrib * m_attrib;
	int m_glyphType;
	
public:
	enum { Type = UserType + 1 };
	
	GardenGlyph(const QPixmap & iconPix,
			QGraphicsItem * parent = 0 );
			
	void setAttrib(PieceAttrib * attrib);
	PieceAttrib * attrib();
	PieceAttrib * attrib() const;
	int attribInstanceId() const;
	
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

/// before (dis)connect with another via port	
	void postConnection(GardenGlyph* another, aphid::GlyphPort* viaPort);
	void preDisconnection(GardenGlyph* another, aphid::GlyphPort* viaPort);
/// after connection via port is changed
	void postDisconnection(aphid::GlyphPort* viaPort);
	void postSelection();
	
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