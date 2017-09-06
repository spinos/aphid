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
#include "data/billboard.h"
#include "data/variation.h"
#include "data/file.h"
#include "data/stem.h"
#include "data/twig.h"
#include "data/branch.h"
#include "data/trunk.h"

PiecesList::PiecesList(QWidget *parent)
    : QListWidget(parent)
{
    setDragEnabled(true);
    setViewMode(QListView::IconMode);
    setIconSize(QSize(32, 32));
	setGridSize(QSize(36, 36));
    setSpacing(2);
    setAcceptDrops(false);
    setDropIndicatorShown(false);
	setMaximumHeight(48);
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

void PiecesList::showSpritePieces()
{
	const int gbegin = gar::GlyphRange[gar::ggSprite][0];
	const int gend = gar::GlyphRange[gar::ggSprite][1];
	lsPieces(gbegin, gend, gar::ggSprite,
			gar::BillboardTypeIcons);
}

void PiecesList::showVariationPieces()
{
	const int gbegin = gar::GlyphRange[gar::ggVariant][0];
	const int gend = gar::GlyphRange[gar::ggVariant][1];
	lsPieces(gbegin, gend, gar::ggVariant,
			gar::VariationTypeIcons);
}

void PiecesList::showStemPieces()
{
	const int gbegin = gar::GlyphRange[gar::ggStem][0];
	const int gend = gar::GlyphRange[gar::ggStem][1];
	lsPieces(gbegin, gend, gar::ggStem,
			gar::StemTypeIcons);
}

void PiecesList::showTwigPieces()
{
	const int gbegin = gar::GlyphRange[gar::ggTwig][0];
	const int gend = gar::GlyphRange[gar::ggTwig][1];
	lsPieces(gbegin, gend, gar::ggTwig,
			gar::TwigTypeIcons);
}

void PiecesList::showBranchPieces()
{
	const int gbegin = gar::GlyphRange[gar::ggBranch][0];
	const int gend = gar::GlyphRange[gar::ggBranch][1];
	lsPieces(gbegin, gend, gar::ggBranch,
			gar::BranchTypeIcons);
}

void PiecesList::showTrunkPieces()
{
	const int gbegin = gar::GlyphRange[gar::ggTrunk][0];
	const int gend = gar::GlyphRange[gar::ggTrunk][1];
	lsPieces(gbegin, gend, gar::ggTrunk,
			gar::TrunkTypeIcons);
}

void PiecesList::lsPieces(const int & gbegin,
				const int & gend,
				const int & ggroup,
				const char * iconNames[])
{
	QListWidget::clear();
	for(int i=gbegin;i<gend;++i) {
		QListWidgetItem *pieceItem = new QListWidgetItem(this);
	
		QPixmap pixm(iconNames[i - gar::ToGroupBegin(ggroup)]);
		QIcon cloverIcon(pixm);
		pieceItem->setIcon(cloverIcon);
		pieceItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable
							| Qt::ItemIsDragEnabled);
		pieceItem->setData(Qt::UserRole, QVariant(pixm) );
		QPoint tg(i, ggroup);
		pieceItem->setData(Qt::UserRole+1, tg);
	}

}
