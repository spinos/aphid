#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <app_define.h>

class BvhSolver;
class BaseBuffer;
class SimpleMesh;
class RayTest;
class CudaParticleSystem;
class TetrahedronSystem;

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
	void debugDraw(unsigned rootInd, unsigned numInternal);
	void drawParticles();
	void drawMesh();
	void drawTetra();
private:
	SimpleMesh * m_mesh;
	RayTest * m_ray;
	CudaParticleSystem * m_particles;
	TetrahedronSystem * m_tetra;
	BvhSolver * m_solver;
	int m_displayLevel;
	
#ifdef BVHSOLVER_DBG_DRAW
	BaseBuffer * m_displayLeafAabbs;
	BaseBuffer * m_displayInternalAabbs;
	BaseBuffer * m_displayInternalDistance;
	BaseBuffer * m_displayLeafHash;
	BaseBuffer * m_internalChildIndices;
	int * m_rootNodeInd;
#endif

private slots:
    
};
//! [3]

#endif
