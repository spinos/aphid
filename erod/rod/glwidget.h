#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <qt/Base3DView.h>
#include <pbd/pbd_common.h>

namespace aphid {
class RotationHandle;

template<typename T>
class GenericHexahedronGrid;
 
}

class SolverThread;
class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:

    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
protected:
	virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    virtual void resetPerspViewTransform();
	virtual void resetOrthoViewTransform();
	
private:
	void drawWindTurbine();
	void drawMesh(const int& nind, const int* inds, const float* pos, const float* nml);
	void addWindSpeed(float x);
	void drawGrid();
	void shuffleSample();
	void makeDynCache();
	void beginCaching();
	
private:
    SolverThread * m_solver;
	aphid::RotationHandle * m_roth;
	aphid::Vector3F m_smpV;
	
	enum WorkMode {
	    wmInteractive = 0,
	    wmMakeingCache = 1
	};
	WorkMode m_workMode;
	
	
typedef aphid::GenericHexahedronGrid<float> GridTyp;
	GridTyp * m_grd;
	
signals:
    void sendBeginCache();
    
private slots:
    void recvEndCache();
    
};
//! [3]

#endif
