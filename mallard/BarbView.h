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
	void receiveSeed(int s);
	void receiveNumSeparate(int n);
	void receiveSeparateStrength(double k);
	void receiveFuzzy(double f);
private:

private:
	unsigned m_numLines;
	unsigned * m_numVerticesPerLine;
	Vector3F * m_vertices;
	Vector3F * m_colors;
	unsigned m_seed;
	unsigned m_numSeparate;
	float m_separateStrength;
	float m_fuzzy;
};
