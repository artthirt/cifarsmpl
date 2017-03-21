#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

const QString model_file("model.bin");

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	m_doIter = false;
	m_batch = 50;
	m_delimiter = 50;
	m_delay = 5;
	m_wid = 0;

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(m_delay);

	ui->wcifar->setCifar(&m_cifar);

	std::vector< ct::ParamsCnv > cnv;
	std::vector< ct::ParamsMlp > mlp;

	ct::generator.seed(17);

	cnv.push_back(ct::ParamsCnv(3, 64, true, 0.93, 0.0001));
	cnv.push_back(ct::ParamsCnv(3, 128, true, 0.92, 0.0001));
	cnv.push_back(ct::ParamsCnv(3, 256, false, 0.93, 0.0001));
	cnv.push_back(ct::ParamsCnv(3, 512, false, 0.92, 0.0001));

	mlp.push_back(ct::ParamsMlp(512, 0.95, 0.001));
//	mlp.push_back(ct::ParamsMlp(512, 0.93, 0.001));
//	mlp.push_back(ct::ParamsMlp(512, 0.98, 0.001));
//	mlp.push_back(ct::ParamsMlp(512, 0.92, 0.005));
//	mlp.push_back(ct::ParamsMlp(512, 0.92, 0.005));
//	mlp.push_back(ct::ParamsMlp(128, 1));
	mlp.push_back(ct::ParamsMlp(10, 1));

	m_train.setCifar(&m_cifar);
	m_train.setConvLayers(cnv, ct::Size(32, 32));
	m_train.setMlpLayers(mlp);

	m_train.setUseRandData(true);
	m_train.setRandData(5, 0, 0.1);

	m_train.init();

	ui->tb_main->setCurrentIndex(0);

	ui->pte_logs->appendPlainText("Current directory: " + m_cifar.currentDirectory());
	ui->pte_logs->appendPlainText("Binary data exists: " + QString(m_cifar.isBinDataExists()? "True" : "False"));

	ui->pte_logs->appendPlainText(QString("count of input to MLP %1").arg(m_train.inputToMlp()));

	ui->sb_batch->setValue(m_batch);
	ui->sb_delay->setValue(m_delay);
	ui->sb_iter_numb->setValue(m_delimiter);

//	ui->dsb_dropoutprob->setValue(m_train.dropoutProb());

	ui->sb_wid->setMaximum((int)cnv.size());
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
	if(!m_cifar.isBinDataExists()){
		ui->pte_logs->appendPlainText("<<<Not exists binary data!>>>");
		return;
	}

	if(ui->pb_pass->isChecked()){
		ui->pte_logs->appendPlainText("<<<Not run test when pass work!>>>");
		return;
	}
	double acc = -1, l2 = -1;
	m_train.getEstimateTest(acc, l2, ui->chb_gpu->isChecked());

	ui->lb_out->setText(QString("Test: acc=%1, l2=%2").arg(acc).arg(l2));
	ui->pte_logs->appendPlainText(QString("Test: acc=%1, l2=%2").arg(acc).arg(l2));

	update_statistics();
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

	m_train.pass(m_batch, ui->chb_gpu->isChecked());

	uint it = ui->chb_gpu->isChecked()? m_train.iteration_gpu() : m_train.iteration();

	ui->lb_it->setText(QString("Iteraion %1").arg(it));

	if((it % m_delimiter) == 0){
		if(ui->wcifar->mode() == WidgetCifar::TRAIN)
			m_train.getEstimate(800, acc, l2, ui->chb_gpu->isChecked());
		else{
			m_train.getEstimateTest(800, acc, l2, ui->chb_gpu->isChecked());
		}

		qDebug("iteration %d: acc=%f, l2=%f", it, acc, l2);

		ui->lb_out->setText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));
		ui->pte_logs->appendPlainText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));

		update_prediction();
		update_statistics();
	}
}

void MainWindow::update_prediction()
{
	QVector<int> pr = m_train.predict(ui->wcifar->output_data(), ui->chb_gpu->isChecked());

	ui->wcifar->updatePredictfromIndex(pr);

	bool b = ui->chb_gpu->isChecked();
	ui->wdgW->setMat(m_train.cnvW(m_wid, b),
					 m_train.szW(m_wid, b),
					 m_train.Kernels(m_wid, b),
					 m_train.channels(m_wid, b));
}

void MainWindow::update_statistics()
{
	QString stat ="<b>Statistics:</b><br>";
	stat += "<tr> <td><b>y/p</b></td>\t";
	for(int i = 0; i < 10; ++i){
		stat += "<td><b>" + QString::number(i + 1) + "</b></td>";
	}
	stat += "</tr>";
	for(int i = 0; i < 10; ++i){
		stat += "<tr><td><b>" + QString::number(i + 1) + ":</b></td>";
		for(int j = 0; j < 10; ++j){
			double p = m_train.statistics(i, j);
			if(i == j){
				stat += "<td><b>" + QString::number(p, 'f', 2) + "</b></td>";
			}else{
				stat += "<td>" + QString::number(p, 'f', 2) + "</td>";
			}
		}
		stat += "</tr>";
	}
	ui->lb_stat->setText(stat);
}

void MainWindow::on_pb_pass_clicked(bool checked)
{
	if(!m_cifar.isBinDataExists()){
		checked = false;
		ui->pb_pass->setChecked(false);
	}
	m_doIter = checked;
}

void MainWindow::on_chb_gpu_clicked(bool checked)
{
	if(checked && m_cifar.isBinDataExists()){
		m_train.init_gpu();
		ui->pte_logs->appendPlainText(QString("GPU: count of input to MLP %1").arg(m_train.inputToMlp(true)));
	}
}

void MainWindow::on_dsb_alpha_valueChanged(double arg1)
{
	m_train.setAlpha(arg1);
	m_train.setAlphaCnv(arg1);
}

void MainWindow::on_actOpenDir_triggered()
{
	QFileDialog dlg;
	dlg.setAcceptMode(QFileDialog::AcceptOpen);
	dlg.setFileMode(QFileDialog::Directory);

	if(dlg.exec()){
		if(m_cifar.openDir(dlg.directory().absolutePath())){
			ui->wcifar->update_source();
		}
	}
}

void MainWindow::on_action_save_model_triggered()
{
	m_train.saveToFile(model_file, ui->chb_gpu->isChecked());
	ui->pte_logs->appendPlainText("Model saved");
}

void MainWindow::on_actionLoad_model_triggered()
{
	if(!m_train.loadFromFile(model_file, ui->chb_gpu->isChecked())){
		ui->pte_logs->appendPlainText("Model not loaded");
	}
	ui->pte_logs->appendPlainText("Model loaded");
}

void MainWindow::on_sb_delay_valueChanged(int arg1)
{
	m_delay = arg1;
	m_timer.setInterval(arg1);
}

void MainWindow::on_sb_batch_valueChanged(int arg1)
{
	m_batch = arg1;
}

void MainWindow::on_sb_iter_numb_valueChanged(int arg1)
{
	m_delimiter = arg1;
}

void MainWindow::on_pb_mode_clicked(bool checked)
{
	if(checked)
		ui->wcifar->setTestMode();
	else
		ui->wcifar->setTrainMode();
}

void MainWindow::on_pb_update_clicked()
{
	if(!m_cifar.isBinDataExists()){
		return;
	}

	update_prediction();
	update_statistics();
}

void MainWindow::on_sb_wid_valueChanged(int arg1)
{
	m_wid = arg1;

	bool b = ui->chb_gpu->isChecked();
	ui->wdgW->setMat(m_train.cnvW(m_wid, b),
					 m_train.szW(m_wid, b),
					 m_train.Kernels(m_wid, b),
					 m_train.channels(m_wid, b));
}

void MainWindow::on_dsb_dropoutprob_valueChanged(double arg1)
{
	m_train.setDropoutProb(arg1);
}

void MainWindow::on_actionSave_matrices_triggered()
{
	m_train.save_weights(ui->chb_gpu->isChecked());
}

void MainWindow::on_chb_debug_clicked(bool checked)
{
	m_train.setDebug(checked);
}

void MainWindow::on_dsb_lambda_valueChanged(double arg1)
{
	m_train.setLambdaMlp(arg1);
}
