#include "BarbView.h"
#include <QtGui>
#include <MlFeather.h>
#include <MlFeatherCollection.h>
#include <KdTreeDrawer.h>
#include <PseudoNoise.h>
#include "MlVane.h"
BarbView::BarbView(QWidget *parent) : Base3DView(parent)
{
	std::cout<<"Barbview ";
	m_seed = 99;
	m_numSeparate = 9;
	m_separateStrength = 0.f;
	m_fuzzy = 0.f;
	m_gridShaft = 100;
	m_gridBarb = 10;
	m_numVerticesPerLine = 0;
	m_vertices = 0;
	m_colors = 0;
	createLines();
}

BarbView::~BarbView()
{
    clear();
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
	f->setSeed(m_seed);
	f->setNumSeparate(m_numSeparate);
	f->setSeparateStrength(m_separateStrength);
	f->setFuzzy(m_fuzzy);
	f->setGrid(m_gridShaft, m_gridBarb);
	f->sampleColor(m_gridShaft, m_gridBarb, m_colors);
	f->samplePosition(m_gridShaft, m_gridBarb, m_vertices);

	update();
}

void BarbView::receiveSeed(int s)
{
	m_seed = s;
	receiveShapeChanged();
}

void BarbView::receiveNumSeparate(int n)
{
	m_numSeparate = n;
	receiveShapeChanged();
}

void BarbView::receiveSeparateStrength(double k)
{
	m_separateStrength = k;
	receiveShapeChanged();
}

void BarbView::receiveFuzzy(double f)
{
	m_fuzzy = f;
	receiveShapeChanged();
}

void BarbView::receiveGridShaft(int g)
{
	m_gridShaft = g;
	createLines();
	receiveShapeChanged();
}

void BarbView::receiveGridBarb(int g)
{
	m_gridBarb = g;
	createLines();
	receiveShapeChanged();
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

void BarbView::createLines()
{
	clear();
	m_numLines = (m_gridShaft + 1) * 2;
	m_numVerticesPerLine = new unsigned[m_numLines];
	for(unsigned i = 0; i < m_numLines; i++) m_numVerticesPerLine[i] = m_gridBarb + 1;
	m_vertices = new Vector3F[m_numLines * (m_gridBarb + 1)];
	m_colors = new Vector3F[m_numLines * (m_gridBarb + 1)];
}
