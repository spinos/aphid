#ifndef BASE3DVIEW_H
#define BASE3DVIEW_H


#include <QGLWidget>
#include <BaseCamera.h>

class Base3DView : public QGLWidget
{
    Q_OBJECT

public:
    
    Base3DView(QWidget *parent = 0);
    virtual ~Base3DView();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    
//! [0]

//! [2]
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void processCamera(QMouseEvent *event);
    void processSelection(QMouseEvent *event);
    void processDeselection(QMouseEvent *event);
    void processMouseInput(QMouseEvent *event);
    
    virtual void clientDraw();
    virtual void clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir);
//! [3]
private:
	void updateOrthoProjection();
	void updatePerspProjection();
    QPoint m_lastPos;
    QColor m_backgroundColor;
	BaseCamera* fCamera;
	
	Vector3F m_hitPosition;

};
#endif        //  #ifndef BASE3DVIEW_H

