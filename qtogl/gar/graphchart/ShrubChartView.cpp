/*
 *  ShrubChartView.cpp
 *  garden
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>

#include "ShrubChartView.h"
#include "GardenGlyph.h"
#include <qt/GlyphPort.h>
#include <qt/GlyphConnection.h>
#include <qt/GlyphHalo.h>
#include "gar_common.h"
#include "GlyphBuilder.h"
#include "ShrubScene.h"

using namespace aphid;

ShrubChartView::ShrubChartView(QGraphicsScene * scene, QWidget * parent) : QGraphicsView(scene, parent)
{
	setAcceptDrops(true);
	setMinimumSize(400, 300);
	setRenderHint(QPainter::Antialiasing);
	qDebug()<<"view rect"<<rect();
	setSceneRect(	frameRect() );
}

void ShrubChartView::mousePressEvent(QMouseEvent *event)
{
	switch (event->button())
	{
		case Qt::LeftButton:
			processSelect(event->pos() );
		break;
		case Qt::RightButton:
			qDebug()<<"todo del";
		break;
		default:
		;
	}
}

void ShrubChartView::mouseMoveEvent(QMouseEvent *event)
{
	QPoint mousePos = event->pos();
	QGraphicsItem *item = itemAt(mousePos);
	
	if(m_mode == mMoveItem) {
		QPointF dv(mousePos.x() - m_lastMosePos.x(),
					mousePos.y() - m_lastMosePos.y() );
		GardenGlyph * gl = (GardenGlyph *)m_selectedItem;
		gl->moveBlockBy(dv);
	}
	if(m_mode == mConnectItems) {
		QPointF pf(mousePos.x(), mousePos.y() );
		m_selectedConnection->setPos1(pf );
		m_selectedConnection->updatePath();
	}
	
	m_lastMosePos = mousePos;
}
	
void ShrubChartView::mouseReleaseEvent(QMouseEvent *event)
{
	QPoint mousePos = event->pos();
	QGraphicsItem *item = itemAt(mousePos);
	
	if(m_mode == mMoveItem) {
		QPointF dv(mousePos.x() - m_lastMosePos.x(),
					mousePos.y() - m_lastMosePos.y() );
		GardenGlyph * gl = (GardenGlyph *)m_selectedItem;
		gl->moveBlockBy(dv);
	}
	if(m_mode == mConnectItems) {
		if(isIncomingPort(item)) {
			QPointF pf = item->scenePos();
			m_selectedConnection->setPos1(pf );
			GlyphPort * pt = (GlyphPort *)item;
			m_selectedConnection->setPort1(pt);
			m_selectedConnection->updatePath();
		}
		
		if(!m_selectedConnection->isComplete() ) {
			delete m_selectedConnection;
		}
	}
	m_mode = mNone;
}

void ShrubChartView::dragEnterEvent(QDragEnterEvent *event)
{
	if (event->mimeData()->hasFormat(gar::PieceMimeStr))
        event->accept();
    else
        event->ignore();
}

void ShrubChartView::dragMoveEvent(QDragMoveEvent *event)
{
    if (event->mimeData()->hasFormat(gar::PieceMimeStr)
        //&& findPiece(targetSquare(event->pos())) == -1
		) {

        //highlightedRect = targetSquare(event->pos());
        event->setDropAction(Qt::MoveAction);
        event->accept();
    } else {
        //highlightedRect = QRect();
        event->ignore();
    }

    /*update(updateRect);*/
}

void ShrubChartView::dropEvent(QDropEvent *event)
{
	qDebug()<<"ShrubChartView::dragMoveEvent"<<event->pos();
    
    if (event->mimeData()->hasFormat(gar::PieceMimeStr) ) {

		QByteArray pieceData = event->mimeData()->data(gar::PieceMimeStr);
		QDataStream dataStream(&pieceData, QIODevice::ReadOnly);
		QPixmap pixmap;
		QPoint pieceTypGrp;
		dataStream >> pixmap >> pieceTypGrp;
		
		addGlyphPiece(pieceTypGrp, pixmap, event->pos() );

		event->setDropAction(Qt::MoveAction);
		event->accept();
    } else {
		event->ignore();
	}
	
}

void ShrubChartView::addGlyphPiece(const QPoint & pieceTypGrp, 
						const QPixmap & px,
						const QPoint & pos)
{
	GardenGlyph * g = new GardenGlyph(px);
	scene()->addItem(g);
	
	QPointF posmts = 	mapToScene(pos);
	g->setPos(posmts);
	
	const int & gtype = pieceTypGrp.x();
	const int & ggroup = pieceTypGrp.y();
	
	GlyphBuilder bdr;
	bdr.build(g, gtype, ggroup);
	
	GlyphHalo* hal = new GlyphHalo;
	posmts += g->localCenter();
	hal->setPos(posmts.x() - 50, posmts.y() - 50 );
	scene()->addItem(hal);
	g->setHalo(hal);
}

void ShrubChartView::processSelect(const QPoint & pos)
{
	m_mode = mPanView;
	QGraphicsItem *item = itemAt(pos);
	if (item) {
         if(isOutgoingPort(item) ) {
			m_mode = mConnectItems;
			m_selectedConnection = new GlyphConnection();
			m_selectedConnection->setPos0(item->scenePos() );
			
			GlyphPort * pt = (GlyphPort *)item;
			m_selectedConnection->setPort0(pt);
			
			scene()->addItem(m_selectedConnection);
			
		 } else {
			m_selectedItem = item->topLevelItem();
			if(m_selectedItem->type() == GardenGlyph::Type ) {
				m_mode = mMoveItem;
				GardenGlyph * gl = (GardenGlyph *)m_selectedItem;
				gl->showHalo();
				ShrubScene* ssc = (ShrubScene* )scene();
				ssc->selectGlyph(gl);
				emit sendSelectGlyph(true);
			}
		 }
     } else {
		ShrubScene* ssc = (ShrubScene* )scene();
		ssc->deselectGlyph();
		emit sendSelectGlyph(false);
	 }
	 m_lastMosePos = pos;
}

bool ShrubChartView::isItemPort(const QGraphicsItem *item) const
{
	if(!item) {
		return false;
	}	
	if(item->type() != GlyphPort::Type) {
		return false;
	}
	return true;
}

bool ShrubChartView::isOutgoingPort(const QGraphicsItem *item) const
{
	if(!isItemPort(item) ) {
		return false;
	}
	const GlyphPort * pt = (const GlyphPort *)item;
	return pt->isOutgoing();
}

bool ShrubChartView::isIncomingPort(const QGraphicsItem *item) const
{
	if(!isItemPort(item) ) {
		return false;
	}
	const GlyphPort * pt = (const GlyphPort *)item;
	return !pt->isOutgoing();
}
