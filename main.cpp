#include "mainwindow.h"
#include <QApplication>

#include "test_agg.h"

int main(int argc, char *argv[])
{
	test_agg tagg;
	tagg.test_hconcat();

	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	return a.exec();
}
