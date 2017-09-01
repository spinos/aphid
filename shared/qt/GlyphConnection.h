/*
 *  GlyphConnection.h
 *
 *  from port0 to port1
 *
 *  Created by jian zhang on 4/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GLYPH_CONNECTION_H
#define APH_GLYPH_CONNECTION_H

#include <QGraphicsPathItem>

namespace aphid {

class GlyphPort;

class GlyphConnection : public QGraphicsPathItem
{
public:
	enum { Type = UserType + 3 };
	
	GlyphConnection(QGraphicsItem * parent = 0);
	virtual ~GlyphConnection();
	
	void setPos0(const QPointF & p);
	void setPos1(const QPointF & p);
	void setPort0(GlyphPort * p);
	void setPort1(GlyphPort * p);
	void disconnectPort(GlyphPort * p);
	
	void updatePath();
	void updatePathByPort(GlyphPort * p);
	bool isComplete() const;
	
	const GlyphPort * port0() const;
	const GlyphPort * port1() const;
	
	virtual bool canConnectTo(GlyphPort* p1) const;
	virtual void breakUp();
	
	int type() const { return Type; }
	
	static bool IsItemConnection(const QGraphicsItem *item);
	
protected:
	GlyphPort * port0();
	GlyphPort * port1();

private:
	QPointF m_pos0;
	QPointF m_pos1;
	GlyphPort * m_port0;
	GlyphPort * m_port1;
	
};

}

#endif