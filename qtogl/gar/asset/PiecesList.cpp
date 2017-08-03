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
#include "data/ground.h"
#include "data/grass.h"
#include "data/file.h"

PiecesList::PiecesList(QWidget *parent)
    : QListWidget(parent)
{
    setDragEnabled(true);
    setViewMode(QListView::IconMode);
    setIconSize(QSize(32, 32));
    setSpacing(4);
    setAcceptDrops(false);
    setDropIndicatorShown(false);
	
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
    QPoint pieceTypGrp = item->data(Qt::UserRole+1).toPoint();

    dataStream << pixmap << pieceTypGrp;

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

void PiecesList::showGroundPieces()
{
	const int gbegin = gar::GlyphRange[gar::ggGround][0];
	const int gend = gar::GlyphRange[gar::ggGround][1];
	lsPieces(gbegin, gend, gar::ggGround,
			gar::GroundTypeIcons);
}

void PiecesList::showGrassPieces()
{
	const int gbegin = gar::GlyphRange[gar::ggGrass][0];
	const int gend = gar::GlyphRange[gar::ggGrass][1];
	lsPieces(gbegin, gend, gar::ggGrass,
			gar::GrassTypeIcons);
}

void PiecesList::showFilePieces()
{
	const int gbegin = gar::GlyphRange[gar::ggFile][0];
	const int gend = gar::GlyphRange[gar::ggFile][1];
	lsPieces(gbegin, gend, gar::ggFile,
			gar::FileTypeIcons);
}

void PiecesList::lsPieces(const int & gbegin,
				const int & gend,
				const int & ggroup,
				const char * iconNames[])
{
	QListWidget::clear();
	for(int i=gbegin;i<gend;++i) {
		QListWidgetItem *pieceItem = new QListWidgetItem(this);
	
		QPixmap pixm(iconNames[i - ggroup * 32]);
		QIcon cloverIcon(pixm);
		pieceItem->setIcon(cloverIcon);
		pieceItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable
							| Qt::ItemIsDragEnabled);
		pieceItem->setData(Qt::UserRole, QVariant(pixm) );
		QPoint tg(i, ggroup);
		pieceItem->setData(Qt::UserRole+1, tg);
	}

}
