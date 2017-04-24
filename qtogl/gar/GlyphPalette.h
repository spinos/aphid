/*
 *  GlyphPalette.h
 *  garden
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_GLYPH_PALETTE_H
#define GAR_GLYPH_PALETTE_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLineEdit;
class QListWidget;
class QListWidgetItem;
QT_END_NAMESPACE

namespace aphid {
class IconLine;
class NavigatorWidget;
class ContextIconFrame;
}

class AssetDescription;
class PiecesList;

class GlyphPalette : public QWidget
{
	Q_OBJECT
	
public:
	GlyphPalette(QWidget *parent = 0);
	
protected:

signals:
	void onAssetSel(QPoint);
	
public slots:
	void showNamedPieces(const QString & swhat);
	
private slots:
	void selectAGrass(QListWidgetItem * item);
	
private:
	PiecesList * m_glyphList;
	AssetDescription * m_describ;
	
};
#endif