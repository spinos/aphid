#include "BarbView.h"
#include <QtGui>
#include <MlFeather.h>
#include <MlFeatherCollection.h>
#include <KdTreeDrawer.h>
#include "MlVane.h"
BarbView::BarbView(QWidget *parent) : Base3DView(parent)
{
	std::cout<<"Barbview ";
	m_numLines = 202;
	m_numVerticesPerLine = new unsigned[m_numLines];
	for(unsigned i = 0; i < m_numLines; i++) m_numVerticesPerLine[i] = 11;
	m_vertices = new Vector3F[m_numLines * 11];
	m_colors = new Vector3F[m_numLines * 11];
}

BarbView::~BarbView()
{
    delete[] m_numVerticesPerLine;
	delete[] m_vertices;
	delete[] m_colors;
}

void BarbView::clientDraw()
{
    if(!FeatherLibrary) return;
	getDrawer()->lineStripes(m_numLines, m_numVerticesPerLine, m_vertices, m_colors);
}

void BarbView::clientSelect()
{

}

void BarbView::clientMouseInput()
{
	
}

void BarbView::receiveShapeChanged()
{
	MlFeather *f = FeatherLibrary->selectedFeatherExample();
    if(!f) return;
	
	float * dst = f->angles();
	const short ns = f->numSegment();
	for(short s=0; s < ns; s++) {
		dst[s] = -0.4f * s / (float)ns;
	}
	f->bend();
	f->updateVane();
    
    MlVane * vane = f->vane(0);
	float du = 1.f/(float)(m_numLines/2 - 1);
	float dv = 1.f/10.f;
	unsigned acc = 0;
	for(unsigned i = 0; i < m_numLines/2; i++) {
		vane->setU(du*i);
		for(unsigned j = 0; j <= 10; j++) {
			vane->pointOnVane(dv * j, m_vertices[acc]);
			acc++;
		}
	}
	vane = f->vane(1);
	for(unsigned i = 0; i < m_numLines/2; i++) {
		vane->setU(du*i);
		for(unsigned j = 0; j <= 10; j++) {
			vane->pointOnVane(dv * j, m_vertices[acc]);
			acc++;
		}
	}
	
	f->sampleColor(100, 10, m_colors);
	update();
}
