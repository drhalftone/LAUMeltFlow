#include "MainWindow.h"
#include "ChainSimulator.h"
#include "ChainWidget.h"
#include "Bead.h"

#include <QStatusBar>
#include <QVector2D>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("Bead Chain Simulation");
    resize(800, 700);

    // ── Build the simulation ───────────────────────────────────────
    m_sim = new ChainSimulator(this);
    m_sim->buildChain(
        /*nBeads=*/    20,
        /*totalLength=*/2.0,
        /*totalMass=*/ 0.5,
        /*stiffness=*/ 10000.0,
        /*damping=*/   2.0,     // ~6% of critical — damps spring buzz
        /*drag=*/      0.02,    // very light air resistance
        /*anchorPos=*/ QVector2D(0.0f, 0.0f)
    );

    // ── Rendering widget ───────────────────────────────────────────
    m_view = new ChainWidget(m_sim, this);
    setCentralWidget(m_view);

    connect(m_sim, &ChainSimulator::stepped,
            m_view, &ChainWidget::onSimStepped);

    // ── Timer drives the simulation at ~60 fps ─────────────────────
    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &MainWindow::tick);
    m_timer->start(16); // ~60 Hz

    statusBar()->showMessage("Simulation running");
}

void MainWindow::tick()
{
    // Run multiple sub-steps per frame to keep physics stable
    // while rendering at 60 fps.
    // 100 sub-steps * 0.0001s = 0.01s of sim time per frame
    // → simulation runs at ~0.6x real time at 60 fps
    for (int i = 0; i < m_subStepsPerFrame; ++i) {
        m_sim->step(m_dt, m_gravity);
        m_simTime += m_dt;
    }

    statusBar()->showMessage(
        QString("t = %1 s").arg(m_simTime, 0, 'f', 3));
}
