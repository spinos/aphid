#include <QApplication>

#include "changableButton.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    ChangableButton cgb;
    cgb.show();
    return app.exec();
}
