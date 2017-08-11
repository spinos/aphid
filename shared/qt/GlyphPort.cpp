/*
 *  GlyphPort.cpp
 *  
 *
 *  Created by jian zhang on 4/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "GlyphPort.h"
#include "GlyphConnection.h"

namespace aphid {

GlyphPort::GlyphPort(QGraphicsItem * parent) : QGraphicsEllipseItem(parent)
{
	setRect(-7, -7, 14, 14);
	setPen(QPen(Qt::darkGray));
	setBrush(Qt::lightGray);
	setZValue(2);
}

GlyphPort::~GlyphPort()
{
	foreach(GlyphConnection *conn, m_connections) {
		conn->disconnectPort(this);
	}
}

void GlyphPort::setPortName(const QString & name)
{ m_portName = name; }

void GlyphPort::setIsOutgoing(bool x)
{ m_isOutgoing = x; }

const QString & GlyphPort::portName() const
{ return m_portName; }

const bool & GlyphPort::isOutgoing() const
{ return m_isOutgoing; }

void GlyphPort::addConnection(GlyphConnection * conn)
{   
    m_connections.append(conn);
}

void GlyphPort::removeConnection(GlyphConnection * conn2rm)
{
	int found = -1;
	int i=0;
	foreach(GlyphConnection *conn, m_connections) {
		if(conn == conn2rm) {
			found = i;
		}
		i++;
	}
	if(found > -1) {
		m_connections.remove(found);
	}
}

void GlyphPort::updateConnectionsPath()
{
	foreach(GlyphConnection *conn, m_connections) {
		conn->updatePathByPort(this);
	}
}

int GlyphPort::numConnections() const
{
	return m_connections.size();
}

const GlyphConnection * GlyphPort::connection(const int & i) const
{ return m_connections[i]; }

}
