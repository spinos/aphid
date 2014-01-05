#pragma once
#include <Base3DView.h>
#include <FeatherExample.h>
#include <LODFn.h>

class BarbView : public Base3DView, public FeatherExample, public LODFn {
Q_OBJECT

public:
    BarbView(QWidget *parent = 0);
    ~BarbView();
	
protected:
	virtual void clientSelect();
	virtual void clientDeselect();
	virtual void clientDraw();
	virtual void clientMouseInput();
	virtual void focusInEvent(QFocusEvent * event);
	
public slots:
	void receiveShapeChanged();
	void receiveLodChanged(double l);
	void test();
private:
};
