#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	ui->wcifar->setCifar(&m_cifar);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_pb_prev_clicked()
{
	ui->wcifar->prev();
}

void MainWindow::on_pb_next_clicked()
{
	ui->wcifar->next();
}
