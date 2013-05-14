#pragma once

#include <Base3DView.h>
class KnitPatch;
class PatchMesh;
class KnitView : public Base3DView
{
    Q_OBJECT

public:
    KnitView(QWidget *parent = 0);
    ~KnitView();

//! [2]
protected:
    virtual void clientDraw();

//! [3]
private:
    unsigned biggestDu(float *u) const;	
	KnitPatch* m_knit;
	PatchMesh * _model;
};
