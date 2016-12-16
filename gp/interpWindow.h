#ifndef GP_INTERP_WINDOW_H
#define GP_INTERP_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class InterpWidget;
   
class InterpWindow : public QMainWindow
{
    Q_OBJECT

public:
    InterpWindow();
	virtual ~InterpWindow();
	
protected:
    void keyPressEvent(QKeyEvent *event);
	
private:
	
private slots:
	
private:
	InterpWidget * m_wig;
	
};
#endif

