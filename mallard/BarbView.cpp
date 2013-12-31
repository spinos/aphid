#include "BarbView.h"
#include <QtGui>
#include <MlFeather.h>
#include <MlFeatherCollection.h>
#include <KdTreeDrawer.h>
#include <PseudoNoise.h>
#include "MlVane.h"
BarbView::BarbView(QWidget *parent) : Base3DView(parent)
{
	std::cout<<" Barbview ";
	m_gridShaft = 0;
	m_gridBarb = 0;
	m_numLines = 0;
	m_numVerticesPerLine = 0;
	m_vertices = 0;
	m_colors = 0;
}

BarbView::~BarbView()
{
    clear();
}

void BarbView::clientDraw()
{
    if(m_numLines < 1) return;
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
	MlFeather *f = selectedExample();
	if(!f) return;
	
	createLines(f->gridShaft(), f->gridBarb());
	
	float * dst = f->angles();
	const short ns = f->numSegment();
	for(short s=0; s < ns; s++) {
		dst[s] = -0.4f * s / (float)ns;
	}
	f->bend();
	f->testVane();
	
	f->sampleColor(f->gridShaft(), f->gridBarb(), m_colors);
	f->samplePosition(m_vertices);

	update();
}

void BarbView::clear()
{
	if(m_numVerticesPerLine) delete[] m_numVerticesPerLine;
	if(m_vertices) delete[] m_vertices;
	if(m_colors) delete[] m_colors;
	m_numVerticesPerLine = 0;
	m_vertices = 0;
	m_colors = 0;
}

void BarbView::createLines(unsigned gridShaft, unsigned gridBarb)
{
	if(gridShaft == m_gridShaft && gridBarb == m_gridBarb) return;
	
	clear();
	
	m_gridShaft = gridShaft;
	m_gridBarb = gridBarb;
	m_numLines = (m_gridShaft + 1) * 2;
	m_numVerticesPerLine = new unsigned[m_numLines];
	for(unsigned i = 0; i < m_numLines; i++) m_numVerticesPerLine[i] = m_gridBarb + 1;
	m_vertices = new Vector3F[m_numLines * (m_gridBarb + 1)];
	m_colors = new Vector3F[m_numLines * (m_gridBarb + 1)];
}
