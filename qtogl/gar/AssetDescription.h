/*
 *  AssetDescription.h
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_ASSET_DESCRIPTION_H
#define GAR_ASSET_DESCRIPTION_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
QT_END_NAMESPACE

class AssetDescription : public QWidget
{
	Q_OBJECT
	
public:
	AssetDescription(QWidget *parent = 0);
	
protected:

signals:
	
public slots:
	void recvAssetSel(int x);
	
private slots:
	
private:
	QLabel * m_lab;
	QLabel * m_pic;
	QLabel * m_dtl;
	
};

#endif