#include "BarbView.h"
#include <QtGui>
#include <MlFeather.h>
#include <MlFeatherCollection.h>
#include <KdTreeDrawer.h>
#include "MlVane.h"
BarbView::BarbView(QWidget *parent) : Base3DView(parent)
{
	std::cout<<"Barbview ";
	m_numLines = 200;
	m_numVerticesPerLine = new unsigned[m_numLines];
	for(unsigned i = 0; i < m_numLines; i++) m_numVerticesPerLine[i] = 17;
	m_vertices = new Vector3F[m_numLines * 17];
}

BarbView::~BarbView()
{
    
}

void BarbView::clientDraw()
{
    if(!FeatherLibrary) return;
    MlFeather *f = FeatherLibrary->selectedFeatherExample();
    if(!f) return;
    
    MlVane * vane = f->vane(0);
	float du = 1.f/(float)m_numLines*2.f;
	float dv = 1.f/16.f;
	unsigned acc = 0;
	for(unsigned i = 0; i <= m_numLines/2; i++) {
		vane->setU(du*i);
		for(unsigned j = 0; j <= 16; j++) {
			vane->pointOnVane(dv * j, m_vertices[acc]);
			acc++;
		}
	}
	vane = f->vane(1);
	for(unsigned i = 0; i <= m_numLines/2; i++) {
		vane->setU(du*i);
		for(unsigned j = 0; j <= 16; j++) {
			vane->pointOnVane(dv * j, m_vertices[acc]);
			acc++;
		}
	}
	getDrawer()->lineStripes(m_numLines, m_numVerticesPerLine, m_vertices);
}

void BarbView::clientSelect()
{

}

void BarbView::clientMouseInput()
{
	
}
