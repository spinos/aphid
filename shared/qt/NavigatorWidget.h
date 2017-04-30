/*
 *   NavigatorWidget.h
 *
 *   image preview
 */
 
#ifndef APH_NAVIGATOR_WIDGET_H
#define APH_NAVIGATOR_WIDGET_H

#include <qt/Plot2DWidget.h>

namespace aphid {

class ExrImage;

class NavigatorWidget : public aphid::Plot2DWidget {

	Q_OBJECT
	
public:
	NavigatorWidget(const ExrImage * img,
				QWidget *parent = 0);
	virtual ~NavigatorWidget();
	
protected:
	virtual void processCamera(QMouseEvent *event);
	
private:

};

}
#endif
