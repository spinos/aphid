#include "BarbView.h"
#include <QtGui>
#include <MlFeather.h>
#include <MlFeatherCollection.h>
#include <KdTreeDrawer.h>
BarbView::BarbView(QWidget *parent) : Base3DView(parent)
{
	std::cout<<"Barbview ";
}

BarbView::~BarbView()
{
    
}

void BarbView::clientDraw()
{
    if(!FeatherLibrary) return;
    MlFeather *f = FeatherLibrary->selectedFeatherExample();
    if(!f) return;
    
    getDrawer()->boundingRectangle(f->getBoundingRectangle());
	
}

void BarbView::clientSelect()
{

}

void BarbView::clientMouseInput()
{
	
}
