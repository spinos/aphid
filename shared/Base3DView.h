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
class GLHUD;

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
	
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    
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
	void receiveBrushChanged();
    
protected:
	virtual void processSelection(QMouseEvent *event);
    virtual void processDeselection(QMouseEvent *event);
    virtual void processMouseInput(QMouseEvent *event);
	virtual void processCamera(QMouseEvent *event);
	virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
    virtual Vector3F sceneCenter() const;
    virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
	virtual void focusInEvent(QFocusEvent * event);
	virtual void focusOutEvent(QFocusEvent * event);
	virtual void clearSelection();
	void showBrush() const;
	const Vector3F strokeVector(const float & depth) const;
	
	void hudText(const std::string & t, const int & row) const;
	
private:
	void computeIncidentRay(int x, int y);
	
private:
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
	int m_dx, m_dy;
	char m_isFocused;
	GLHUD * m_hud;
};
#endif        //  #ifndef BASE3DVIEW_H

