#ifndef HERMITE_WIDGET_H
#define HERMITE_WIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

class Vector3F;
class Matrix44F;
class TranslationHandle;
class Ray;

namespace pbd {
class Beam;
}

}

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
	
public slots:
	
signals:

private:
	void selectSeg(int d);
	void toggleTngSel();
	void movePnt();
	void moveTng();
	void printSegs();
	
private:
	aphid::TranslationHandle * m_tranh;
	aphid::Matrix44F* m_space;
	aphid::Ray* m_incident;
	aphid::pbd::Beam* m_beam;
/// [0:3] 3 is end of 3rd seg
	int m_pntI;
	bool m_tngSel;
	
};

#endif
