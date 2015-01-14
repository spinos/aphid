#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <app_define.h>

class BvhSolver;
class BaseBuffer;
class SimpleMesh;

class GLWidget : public Base3DView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
public slots:
	
signals:

protected:
    virtual void clientInit();
    virtual void clientDraw();
	virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
private:
	void showEdgeContacts();
	void debugDraw();
	
	const unsigned numEdges() const;

private:
	SimpleMesh * m_mesh;
	BvhSolver * m_solver;
	BaseBuffer * m_displayRays;
	BaseBuffer * m_edges;
	int m_displayLevel;
	
#ifdef BVHSOLVER_DBG_DRAW
	BaseBuffer * m_displayLeafAabbs;
	BaseBuffer * m_displayInternalAabbs;
	BaseBuffer * m_displayInternalDistance;
	BaseBuffer * m_displayLeafHash;
	BaseBuffer * m_internalChildIndices;
#endif

private slots:
    
};
//! [3]

#endif
