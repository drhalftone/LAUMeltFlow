#include "MainWindow.h"
#include "ChainSimulator.h"
#include "ChainWidget.h"
#include "Bead.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QStatusBar>
#include <QFileDialog>
#include <QFile>
#include <QMessageBox>
#include <QVector2D>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("Bead Chain Simulation");
    resize(1050, 700);

    m_sim = new ChainSimulator(this);

    // -- Central layout: [view | panel] --
    auto *central = new QWidget(this);
    auto *hbox = new QHBoxLayout(central);

    m_view = new ChainWidget(m_sim, central);
    hbox->addWidget(m_view, 1);

    auto *panel = new QWidget(central);
    panel->setFixedWidth(250);
    auto *panelLayout = new QVBoxLayout(panel);

    // -- Parameters group --
    auto *paramGroup = new QGroupBox("Parameters", panel);
    auto *form = new QFormLayout(paramGroup);

    m_spinBeads = new QSpinBox;
    m_spinBeads->setRange(3, 200);
    m_spinBeads->setValue(20);
    form->addRow("Beads:", m_spinBeads);

    m_spinLength = new QDoubleSpinBox;
    m_spinLength->setRange(0.1, 50.0);
    m_spinLength->setValue(2.0);
    m_spinLength->setDecimals(2);
    m_spinLength->setSuffix(" m");
    form->addRow("Length:", m_spinLength);

    m_spinMass = new QDoubleSpinBox;
    m_spinMass->setRange(0.01, 100.0);
    m_spinMass->setValue(0.5);
    m_spinMass->setDecimals(3);
    m_spinMass->setSuffix(" kg");
    form->addRow("Mass:", m_spinMass);

    m_spinStiffness = new QDoubleSpinBox;
    m_spinStiffness->setRange(100, 1e6);
    m_spinStiffness->setValue(10000);
    m_spinStiffness->setDecimals(0);
    m_spinStiffness->setSuffix(" N/m");
    m_spinStiffness->setSingleStep(1000);
    form->addRow("Stiffness:", m_spinStiffness);

    m_spinDamping = new QDoubleSpinBox;
    m_spinDamping->setRange(0.0, 100.0);
    m_spinDamping->setValue(2.0);
    m_spinDamping->setDecimals(2);
    form->addRow("Damping:", m_spinDamping);

    m_spinDrag = new QDoubleSpinBox;
    m_spinDrag->setRange(0.0, 10.0);
    m_spinDrag->setValue(0.02);
    m_spinDrag->setDecimals(3);
    m_spinDrag->setSingleStep(0.01);
    form->addRow("Drag:", m_spinDrag);

    m_spinGravity = new QDoubleSpinBox;
    m_spinGravity->setRange(0.0, 100.0);
    m_spinGravity->setValue(9.81);
    m_spinGravity->setDecimals(2);
    m_spinGravity->setSuffix(" m/s\u00B2");
    form->addRow("Gravity:", m_spinGravity);

    m_spinDt = new QDoubleSpinBox;
    m_spinDt->setRange(0.00001, 0.01);
    m_spinDt->setValue(0.0001);
    m_spinDt->setDecimals(5);
    m_spinDt->setSingleStep(0.00005);
    m_spinDt->setSuffix(" s");
    form->addRow("dt:", m_spinDt);

    m_spinSubSteps = new QSpinBox;
    m_spinSubSteps->setRange(1, 1000);
    m_spinSubSteps->setValue(500);
    form->addRow("Sub-steps/frame:", m_spinSubSteps);

    panelLayout->addWidget(paramGroup);

    // -- Recording group --
    auto *recGroup = new QGroupBox("Recording", panel);
    auto *recForm = new QFormLayout(recGroup);

    m_spinRecordInterval = new QSpinBox;
    m_spinRecordInterval->setRange(1, 1000);
    m_spinRecordInterval->setValue(10);
    m_spinRecordInterval->setToolTip("Record a frame every N sim steps");
    recForm->addRow("Record every:", m_spinRecordInterval);

    m_lblFrames = new QLabel("--");
    recForm->addRow("Recorded:", m_lblFrames);

    panelLayout->addWidget(recGroup);

    // -- Buttons --
    panelLayout->addStretch();

    m_btnRecord = new QPushButton("Record");
    m_btnRecord->setMinimumHeight(36);
    m_btnRecord->setToolTip("Restart simulation with recording, then export");
    panelLayout->addWidget(m_btnRecord);

    m_btnStartReset = new QPushButton("Start");
    m_btnStartReset->setMinimumHeight(40);
    panelLayout->addWidget(m_btnStartReset);

    hbox->addWidget(panel);
    setCentralWidget(central);

    // -- Connections --
    connect(m_btnStartReset, &QPushButton::clicked,
            this, &MainWindow::onStartReset);
    connect(m_btnRecord, &QPushButton::clicked,
            this, &MainWindow::onRecord);
    connect(m_sim, &ChainSimulator::stepped,
            m_view, &ChainWidget::onSimStepped);

    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &MainWindow::tick);

    // Build the initial chain so beads are visible before Start
    m_sim->buildChain(
        m_spinBeads->value(),
        m_spinLength->value(),
        m_spinMass->value(),
        m_spinStiffness->value(),
        m_spinDamping->value(),
        m_spinDrag->value()
    );

    statusBar()->showMessage("Ready");
}

void MainWindow::setParamsEnabled(bool on)
{
    m_spinBeads->setEnabled(on);
    m_spinLength->setEnabled(on);
    m_spinMass->setEnabled(on);
    m_spinStiffness->setEnabled(on);
    m_spinDamping->setEnabled(on);
    m_spinDrag->setEnabled(on);
    m_spinGravity->setEnabled(on);
    m_spinDt->setEnabled(on);
    m_spinSubSteps->setEnabled(on);
    m_spinRecordInterval->setEnabled(on);
}

void MainWindow::buildAndStart(bool recording)
{
    m_sim->buildChain(
        m_spinBeads->value(),
        m_spinLength->value(),
        m_spinMass->value(),
        m_spinStiffness->value(),
        m_spinDamping->value(),
        m_spinDrag->value()
    );

    m_sim->setRecording(recording);
    m_sim->clearRecording();
    m_simTime = 0.0;
    m_stepCount = 0;

    if (recording)
        m_sim->recordFrame(0.0);

    m_running = true;
    setParamsEnabled(false);
    m_btnStartReset->setText("Reset");
}

void MainWindow::stopSim()
{
    m_timer->stop();
    m_running = false;
    m_btnStartReset->setText("Start");
    m_btnRecord->setEnabled(true);
    setParamsEnabled(true);
}

void MainWindow::onStartReset()
{
    if (m_running) {
        stopSim();
        // Rebuild chain in initial position and show it
        m_sim->buildChain(
            m_spinBeads->value(),
            m_spinLength->value(),
            m_spinMass->value(),
            m_spinStiffness->value(),
            m_spinDamping->value(),
            m_spinDrag->value()
        );
        m_view->update();
        statusBar()->showMessage("Reset — press Start");
        m_lblFrames->setText("--");
    } else {
        // Normal start — no recording, full speed
        buildAndStart(false);
        m_btnRecord->setEnabled(false);
        m_lblFrames->setText("--");
        m_timer->start(16);
        statusBar()->showMessage("Running (real-time)");
    }
}

void MainWindow::onRecord()
{
    if (m_running)
        return; // shouldn't happen, button is disabled

    // Rebuild chain and run the entire simulation off-screen,
    // recording every N steps, then export.
    buildAndStart(true);
    m_btnRecord->setEnabled(false);
    m_btnStartReset->setEnabled(false);

    double dt = m_spinDt->value();
    double gravity = m_spinGravity->value();
    int recInterval = m_spinRecordInterval->value();

    // Run for a fixed duration (use current dt to determine steps)
    // 5 seconds of simulation
    double duration = 5.0;
    int totalSteps = static_cast<int>(duration / dt);

    statusBar()->showMessage("Recording... please wait");
    QApplication::processEvents();

    for (int i = 1; i <= totalSteps; ++i) {
        m_sim->step(dt, gravity);
        m_simTime += dt;
        m_stepCount++;

        if (m_stepCount % recInterval == 0)
            m_sim->recordFrame(m_simTime);

        // Update UI periodically so it doesn't freeze
        if (i % 5000 == 0) {
            m_lblFrames->setText(QString("%1 frames").arg(
                m_sim->recording().size()));
            statusBar()->showMessage(
                QString("Recording... t = %1 / %2 s")
                .arg(m_simTime, 0, 'f', 2).arg(duration, 0, 'f', 1));
            QApplication::processEvents();
        }
    }

    int nFrames = m_sim->recording().size();
    m_lblFrames->setText(QString("%1 frames").arg(nFrames));
    statusBar()->showMessage(
        QString("Done — %1 frames recorded. Saving...").arg(nFrames));

    // Auto-export
    exportRecording();

    // Reset state
    m_running = false;
    m_btnStartReset->setText("Start");
    m_btnStartReset->setEnabled(true);
    m_btnRecord->setEnabled(true);
    setParamsEnabled(true);

    // Repaint with final state
    m_view->update();
}

void MainWindow::tick()
{
    double dt = m_spinDt->value();
    double gravity = m_spinGravity->value();
    int subSteps = m_spinSubSteps->value();

    for (int i = 0; i < subSteps; ++i) {
        m_sim->step(dt, gravity);
        m_simTime += dt;
    }

    statusBar()->showMessage(
        QString("t = %1 s").arg(m_simTime, 0, 'f', 3));
}

void MainWindow::exportRecording()
{
    const auto &frames = m_sim->recording();
    if (frames.isEmpty())
        return;

    QString path = QFileDialog::getSaveFileName(
        this, "Export Recording", "chain_recording.csv",
        "CSV files (*.csv)");
    if (path.isEmpty())
        return;

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Export Error",
                             "Could not open file for writing.");
        return;
    }

    QTextStream out(&file);
    int nBeads = frames.first().positions.size();

    // Header
    out << "time";
    for (int b = 0; b < nBeads; ++b)
        out << QString(",x%1,y%1,vx%1,vy%1").arg(b);
    out << "\n";

    // Data
    for (const FrameRecord &f : frames) {
        out << QString::number(f.time, 'g', 10);
        for (int b = 0; b < nBeads; ++b) {
            out << "," << QString::number(f.positions[b].x(), 'g', 10)
                << "," << QString::number(f.positions[b].y(), 'g', 10)
                << "," << QString::number(f.velocities[b].x(), 'g', 10)
                << "," << QString::number(f.velocities[b].y(), 'g', 10);
        }
        out << "\n";
    }
    file.close();

    // Metadata sidecar
    QString metaPath = path;
    metaPath.replace(".csv", "_meta.csv");
    QFile metaFile(metaPath);
    if (metaFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream meta(&metaFile);
        meta << "parameter,value\n";
        meta << "n_beads," << nBeads << "\n";
        meta << "total_length," << m_spinLength->value() << "\n";
        meta << "total_mass," << m_spinMass->value() << "\n";
        meta << "stiffness," << m_spinStiffness->value() << "\n";
        meta << "damping," << m_spinDamping->value() << "\n";
        meta << "drag," << m_spinDrag->value() << "\n";
        meta << "gravity," << m_spinGravity->value() << "\n";
        meta << "dt," << QString::number(m_spinDt->value(), 'g', 10) << "\n";
        meta << "record_interval," << m_spinRecordInterval->value() << "\n";
        meta << "n_frames," << frames.size() << "\n";
        metaFile.close();
    }

    QMessageBox::information(this, "Export Complete",
        QString("Exported %1 frames to:\n%2").arg(frames.size()).arg(path));
}
