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
#include "GardenConnection.h"
#include <qt/GlyphPort.h>
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
	setSceneRect(	frameRect() );
}

void ShrubChartView::mousePressEvent(QMouseEvent *event)
{
	if(event->modifiers() == Qt::AltModifier) {
		beginProcessView(event);		
	} else {
		beginProcessItem(event);
	}
}

void ShrubChartView::mouseMoveEvent(QMouseEvent *event)
{
	if(m_mode == mPanView) {
		panView(event);
	} else {
		processItem(event);
	}
}

void ShrubChartView::panView(QMouseEvent *event)
{
	QPoint mousePos = event->pos();
	QPointF dv(mousePos.x() - m_lastMosePos.x(),
					mousePos.y() - m_lastMosePos.y() );
					
	QRectF frm = sceneRect ();
	frm.adjust(-dv.x(), -dv.y(), -dv.x(), -dv.y() );
	m_sceneOrigin.rx() += dv.x();
	m_sceneOrigin.ry() += dv.y();
	setSceneRect(frm); 
	
	m_lastMosePos = mousePos;
}

void ShrubChartView::processItem(QMouseEvent *event)
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
		pf.rx() -= m_sceneOrigin.x();
		pf.ry() -= m_sceneOrigin.y();
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
			if(m_selectedConnection->canConnectTo(pt) ) {
			    m_selectedConnection->setPort1(pt);
			    m_selectedConnection->updatePath();
			    
			    GardenGlyph * srcNode = m_selectedConnection->node0();
			    GardenGlyph * destNode = m_selectedConnection->node1();
			    destNode->postConnection(srcNode);
			    //qDebug()<<" made connection "<<m_selectedConnection->port0()->portName()
			    //    <<" -> "<<pt->portName();
			        
			} else {
			    
			    //qDebug()<<" rejects connection "<<m_selectedConnection->port0()->parentItem()
			    //<<" -> "<<pt->parentItem();
			}
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

void ShrubChartView::resizeEvent ( QResizeEvent * event )
{
	setSceneRect(QRectF(-m_sceneOrigin.x(), -m_sceneOrigin.y(), 
		event->size().width(), event->size().height() ));
	QGraphicsView::resizeEvent(event);
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
	QGraphicsItem *item = itemAt(pos);
	if (item) {
         if(isOutgoingPort(item) ) {
			m_mode = mConnectItems;
			m_selectedConnection = new GardenConnection;
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

void ShrubChartView::beginProcessView(QMouseEvent *event) 
{
	QPoint mousePos = event->pos();
	m_mode = mPanView;
	m_lastMosePos = mousePos;
}

void ShrubChartView::beginProcessItem(QMouseEvent *event) 
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

