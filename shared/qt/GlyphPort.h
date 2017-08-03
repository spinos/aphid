/*
 *  GlyphPort.h
 *
 *  source of destination of glyph connections
 *
 *  Created by jian zhang on 4/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GLYPH_PORT_H
#define APH_GLYPH_PORT_H

#include <QGraphicsEllipseItem>

namespace aphid {

class GlyphConnection;

class GlyphPort : public QGraphicsEllipseItem
{

public:
	enum { Type = UserType + 2 };
	
	GlyphPort(QGraphicsItem * parent = 0 );
	virtual ~GlyphPort();
	
	void addConnection(GlyphConnection * conn);
	void removeConnection(GlyphConnection * conn2rm);
	void updateConnectionsPath();
	
	void setPortName(const QString & name);
	void setIsOutgoing(bool x);
	
	const QString & portName() const;
	const bool & isOutgoing() const;
	
	int numConnections() const;
	const GlyphConnection * connection(const int & i) const;
	
	int type() const { return Type; }
	
protected:
	
private:
	QVector<GlyphConnection*> m_connections;
	QString m_portName;
	bool m_isOutgoing;
	
};

}
#endif