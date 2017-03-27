/*
 *  GrassPalette.h
 *  garden
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_GRASS_PALETTE_H
#define GAR_GRASS_PALETTE_H

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

class GrassPalette : public QWidget
{
	Q_OBJECT
	
public:
	GrassPalette(QWidget *parent = 0);
	
protected:

signals:
	
public slots:
	
private slots:
	void selectAGrass(QListWidgetItem * item);
	
private:
	QListWidget * m_grassList;
	
};
#endif