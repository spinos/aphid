#ifndef BASE3DVIEW_H
#define BASE3DVIEW_H

#include <QGLWidget>
#include <PerspectiveCamera.h>
#include <KdTreeDrawer.h>
#include <SelectionArray.h>
#include <IntersectionContext.h>

class QTimer;

class Base3DView : public QGLWidget
{
    Q_OBJECT

public:
    
    Base3DView(QWidget *parent = 0);
    virtual ~Base3DView();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    
//! [0]

	BaseCamera * getCamera() const;
	KdTreeDrawer * getDrawer() const;
	SelectionArray * getSelection() const;
	IntersectionContext * getIntersectionContext() const;
	
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void processSelection(QMouseEvent *event);
    void processDeselection(QMouseEvent *event);
    void processMouseInput(QMouseEvent *event);
	void processCamera(QMouseEvent *event);
    virtual void clientDraw();
    virtual void clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir);
    virtual void sceneCenter(Vector3F & dst) const;
    
    void resetView();
	void drawSelection();
	void clearSelection();
	void addHitToSelection();
	void growSelection();
	void shrinkSelection();
	void frameAll();
	
protected:
    void keyPressEvent(QKeyEvent *event);
	void focusInEvent(QFocusEvent * event);
	void focusOutEvent(QFocusEvent * event);
private:
	void updateOrthoProjection();
	void updatePerspProjection();
    QPoint m_lastPos;
    QColor m_backgroundColor;
	BaseCamera* fCamera;
	BaseCamera* m_orthoCamera;
	PerspectiveCamera* m_perspCamera;
	KdTreeDrawer * m_drawer;
	SelectionArray * m_selected;
	IntersectionContext * m_intersectCtx;
	QTimer *m_timer;
	char m_isFocused;
};
#endif        //  #ifndef BASE3DVIEW_H

