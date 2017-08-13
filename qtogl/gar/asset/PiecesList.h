/*
 *  PiecesList.h
 *  garden
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_PIECES_LIST_H
#define GAR_PIECES_LIST_H

#include <QListWidget>

class PiecesList : public QListWidget
{
    Q_OBJECT

public:
    PiecesList(QWidget *parent = 0);
	
	void showGrassPieces();
	void showGroundPieces();
	void showFilePieces();
	void showSpritePieces();
	void showVariationPieces();
	void showStemPieces();
	
protected:
	void dragEnterEvent(QDragEnterEvent *event);
    void dragMoveEvent(QDragMoveEvent *event);
    void startDrag(Qt::DropActions supportedActions);
	
private:
	void lsPieces(const int & gbegin,
				const int & gend,
				const int & ggroup,
				const char * iconNames[]);
	
};

#endif