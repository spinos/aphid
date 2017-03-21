/*
 *  HeightFieldAssets.h
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WBG_HEIGHT_FIELD_ASSET_H
#define WBG_HEIGHT_FIELD_ASSET_H

#include <QTreeWidgetItem>

class HeightFieldAssets : public QTreeWidgetItem, public QObject
{
	
public:
	HeightFieldAssets(QTreeWidget *parent = 0);
	
	bool addHeightField(const QString & fileName);
	
protected:

public slots:
	
private slots:

private:

};
#endif