#include <QApplication>
/// #include <time.h> 
#include "drpcawindow.h"

int main(int argc, char *argv[])
{
/// srand(std::time(NULL) );	
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
}
