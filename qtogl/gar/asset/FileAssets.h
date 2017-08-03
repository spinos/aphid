/*
 *  FileAssets.h
 *  garden
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_FILE_ASSETS_H
#define GAR_FILE_ASSETS_H

#include <QTreeWidgetItem>

class FileAssets : public QTreeWidgetItem, public QObject
{
	
public:
	FileAssets(QTreeWidget *parent = 0);
	
protected:

public slots:
	
private slots:

private:

};
#endif