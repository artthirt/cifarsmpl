#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	m_doIter = false;

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(100);

	ui->wcifar->setCifar(&m_cifar);

	std::vector< int > cnv;
	std::vector< int > mlp;
	std::vector< int > cnv_w;

	cnv.push_back(10);
	cnv.push_back(10);

	cnv_w.push_back(5);
	cnv_w.push_back(5);

	mlp.push_back(500);
	mlp.push_back(500);
	mlp.push_back(400);
	mlp.push_back(400);
	mlp.push_back(10);

	m_train.setCifar(&m_cifar);
	m_train.setConvLayers(cnv, cnv_w, ct::Size(32, 32));
	m_train.setMlpLayers(mlp);

	m_train.init();
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

void MainWindow::on_pb_test_clicked()
{
	ct::Matf y;
	std::vector< ct::Matf > X;

	m_cifar.getTrain(100, X, y);

	qDebug("X[0] size=(%d, %d)", X[0].rows, X[0].cols);
}

void MainWindow::onTimeout()
{
	if(m_doIter){
		pass();
	}
}

void MainWindow::on_pb_pass_clicked()
{

}

void MainWindow::pass()
{
	double l2, acc;
	m_train.pass(50);

	if((m_train.iteration() % 10) == 0){
		m_train.getEstimage(300, acc, l2);

		qDebug("iteration %d: acc=%f, l2=%f", m_train.iteration(), acc, l2);

		ui->lb_out->setText(QString("iteration %1: acc=%2, l2=%3").arg(m_train.iteration()).arg(acc).arg(l2));

		update_prediction();
	}
}

void MainWindow::update_prediction()
{
	QVector<int> pr = m_train.predict(ui->wcifar->index(), 500);

	ui->wcifar->updatePredictfromIndex(pr);
}

void MainWindow::on_pb_pass_clicked(bool checked)
{
	m_doIter = checked;
}
