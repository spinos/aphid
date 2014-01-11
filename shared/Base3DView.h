#ifndef BASE3DVIEW_H
#define BASE3DVIEW_H

#include <QGLWidget>
#include <AllMath.h>
#include <Ray.h>
class BaseCamera;
class PerspectiveCamera;
class KdTreeDrawer;
class IntersectionContext;
class ToolContext;
class SelectionArray;
class QTimer;
class BaseBrush;
class BaseTransform;
class TransformManipulator;
class MeshManipulator;

class Base3DView : public QGLWidget
{
    Q_OBJECT

public:
    
    Base3DView(QWidget *parent = 0);
    virtual ~Base3DView();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

	BaseCamera * getCamera() const;
	BaseCamera * perspCamera();
	BaseCamera * orthoCamera();
	KdTreeDrawer * getDrawer() const;
	SelectionArray * getActiveComponent() const;
	IntersectionContext * getIntersectionContext() const;
	const Ray * getIncidentRay() const;
	
	const BaseBrush * brush() const;
	BaseBrush * brush();
	
	TransformManipulator * manipulator();
	MeshManipulator * sculptor();
	
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void processSelection(QMouseEvent *event);
    void processDeselection(QMouseEvent *event);
    void processMouseInput(QMouseEvent *event);
    
    void resetView();
	void drawSelection();
	void addHitToSelection();
	void growSelection();
	void shrinkSelection();
	void frameAll();
	virtual void drawIntersection() const;
	
	void updateOrthoProjection();
	void updatePerspProjection();
	
	QPoint lastMousePos() const;
	
	void setInteractContext(ToolContext * ctx);
	int interactMode() const;
	
	void usePerspCamera();
	void useOrthoCamera();
	
public slots:
	void receiveBrushRadius(double x);
    void receiveBrushPitch(double x);
    void receiveBrushNumSamples(int x);
	void receiveBrushStrength(double x);
	
protected:
	virtual void processCamera(QMouseEvent *event);
    virtual void clientDraw();
    virtual void clientSelect();
    virtual void clientDeselect();
    virtual void clientMouseInput();
    virtual Vector3F sceneCenter() const;
    virtual void keyPressEvent(QKeyEvent *event);
	virtual void focusInEvent(QFocusEvent * event);
	virtual void focusOutEvent(QFocusEvent * event);
	virtual void clearSelection();
	void showBrush() const;
	void showManipulator() const;
private:
	void computeIncidentRay(int x, int y);
	
private:
	TransformManipulator * m_manipulator;
	MeshManipulator * m_sculptor;
	Ray m_incidentRay;
	QPoint m_lastPos;
    QColor m_backgroundColor;
	BaseCamera* fCamera;
	BaseCamera* m_orthoCamera;
	PerspectiveCamera* m_perspCamera;
	KdTreeDrawer * m_drawer;
	SelectionArray * m_activeComponent;
	IntersectionContext * m_intersectCtx;
	BaseBrush * m_brush;
	QTimer *m_timer;
	ToolContext * m_interactContext;
	char m_isFocused;
};
#endif        //  #ifndef BASE3DVIEW_H

