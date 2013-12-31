#pragma once
#include <Base3DView.h>
#include <FeatherExample.h>

class BarbView : public Base3DView, public FeatherExample {
Q_OBJECT

public:
    BarbView(QWidget *parent = 0);
    ~BarbView();
	
	virtual void clientDraw();
	virtual void clientSelect();
	virtual void clientMouseInput();
public slots:
	void receiveShapeChanged();
private:
	void clear();
	void createLines(unsigned gridShaft, unsigned gridBarb);
private:
	unsigned m_numLines, m_gridShaft, m_gridBarb;
	unsigned * m_numVerticesPerLine;
	Vector3F * m_vertices;
	Vector3F * m_colors;
};
