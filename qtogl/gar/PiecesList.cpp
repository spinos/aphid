/*
 *  PiecesList.cpp
 *  garden
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "PiecesList.h"
#include "gar_common.h"
#include "data/grass.h"

PiecesList::PiecesList(QWidget *parent)
    : QListWidget(parent)
{
    setDragEnabled(true);
    setViewMode(QListView::IconMode);
    setIconSize(QSize(32, 32));
    setSpacing(4);
    setAcceptDrops(false);
    setDropIndicatorShown(false);
	
	QListWidgetItem *pieceItem = new QListWidgetItem(this);
	
	QPixmap pixm(gar::GrassTypeIcons[gar::gsClover]);
	QIcon cloverIcon(pixm);
	pieceItem->setIcon(cloverIcon);
	pieceItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable
						| Qt::ItemIsDragEnabled);
	pieceItem->setData(Qt::UserRole, QVariant(pixm) );
	pieceItem->setData(Qt::UserRole+1, gar::gsClover);
    
}

void PiecesList::dragEnterEvent(QDragEnterEvent *event)
{
	qDebug()<<"PiecesList::dragEnterEvent";
	if (event->mimeData()->hasFormat(gar::PieceMimeStr)) {
        event->setDropAction(Qt::MoveAction);
        event->accept();
    } else
        event->ignore();
}

void PiecesList::dragMoveEvent(QDragMoveEvent *event)
{
	qDebug()<<"PiecesList::dragMoveEvent";
	if (event->mimeData()->hasFormat(gar::PieceMimeStr))
        event->accept();
    else
        event->ignore();
}

void PiecesList::startDrag(Qt::DropActions /*supportedActions*/)
{
	QListWidgetItem *item = currentItem();

    QByteArray itemData;
    QDataStream dataStream(&itemData, QIODevice::WriteOnly);
    QPixmap pixmap = qVariantValue<QPixmap>(item->data(Qt::UserRole));
    int pieceTyp = item->data(Qt::UserRole+1).toInt();

    dataStream << pixmap << pieceTyp;

    QMimeData *mimeData = new QMimeData;
    mimeData->setData(gar::PieceMimeStr, itemData);

    QDrag *drag = new QDrag(this);
    drag->setMimeData(mimeData);
    drag->setHotSpot(QPoint(pixmap.width()/2, pixmap.height()/2));
    drag->setPixmap(pixmap);
	qDebug()<<"PiecesList::startDrag"<<dataStream;
	
	drag->exec(Qt::MoveAction);
	setCursor(Qt::OpenHandCursor);
}
