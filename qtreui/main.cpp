#include <QApplication>

#include "RegexUi.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    RegexUi cgb;
    cgb.show();
    return app.exec();
}
