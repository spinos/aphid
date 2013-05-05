/*
 *  TargetView.h
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef TARGETVIEW_H
#define TARGETVIEW_H

#include <SingleModelView.h>

class MeshTopology;

//! [0]
class TargetView : public SingleModelView
{
    Q_OBJECT

public:
    TargetView(QWidget *parent = 0);
    ~TargetView();

	virtual	void buildTree();
	virtual void loadMesh(std::string filename);
    
//! [3]
private:
	MeshTopology *m_topo;

signals:
     void targetChanged();
};
//! [3]

#endif
