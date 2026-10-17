const engine = new Connect4Engine(6, 7);
const brain = new NeuralAgentManager(6, 7);
const visualizer = new NeuralVisualizer('nn-canvas', 'chart-canvas');

// État de l'application
let isTraining = false;
let isArena = false;
let stopRequested = false;
let isProcessingMove = false;
let currentSpeed = '1'; // '1', '5', '10', '20', 'MAX'
let hoveredCol = null;
let lastAITurnContext = null;
let arenaStats = { aWins: 0, bWins: 0, draws: 0 };

// Correspondance entre le sélecteur de vitesse et le délai par pion (en ms)
const SPEED_DELAYS = {
    '1': 550,  // Vitesse humaine : on suit chaque réflexion
    '5': 110,  // Rapide x5
    '10': 45,  // Très rapide x10
    '20': 15,  // Timelapse x20
    'MAX': 0   // Calcul brut en arrière-plan
};

// ==========================================
// 1. INITIALISATION DOM & ONGLETS
// ==========================================

function initDOM() {
    const gridEl = document.getElementById('board');
    gridEl.innerHTML = '';
    for (let r = 0; r < engine.ROWS; r++) {
        for (let c = 0; c < engine.COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell';
            div.id = `cell-${r}-${c}`;
            div.onclick = () => { if (!isArena && !isTraining) handleHumanMove(c); };
            div.onmouseenter = () => { hoveredCol = c; updateGhostPreview(); };
            div.onmouseleave = () => { hoveredCol = null; updateGhostPreview(); };
            gridEl.appendChild(div);
        }
    }

    const brainEl = document.getElementById('brain-bars');
    brainEl.innerHTML = '';
    for (let c = 0; c < engine.COLS; c++) {
        const colDiv = document.createElement('div');
        colDiv.className = 'q-col';
        colDiv.id = `q-col-${c}`;
        colDiv.innerHTML = `
            <div class="q-score" id="q-score-${c}">0.0</div>
            <div class="q-bar-track">
                <div class="q-bar-fill" id="q-fill-${c}"></div>
            </div>
        `;
        brainEl.appendChild(colDiv);
    }

    // Navigation par onglets
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
            visualizer.drawNeuralNetwork(brain.lastInspection);
            visualizer.drawLearningCurve();
        };
    });

    // Sélecteur de vitesse (x1, x5, x10, x20, MAX)
    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSpeed = btn.dataset.speed;
            showPopup(`⏱️ Vitesse : ${currentSpeed === 'MAX' ? '⚡ Turbo MAX' : 'x' + currentSpeed}`, '#70a1ff');
        };
    });

    // Curseurs du Laboratoire (Reward Shaping)
    const bindSlider = (id, labelId, key, suffix = ' XP', scale = 1) => {
        const el = document.getElementById(id);
        const lbl = document.getElementById(labelId);
        if (!el || !lbl) return;
        el.oninput = () => {
            const val = parseFloat(el.value);
            engine.rewardConfig[key] = val * scale;
            lbl.innerText = (key === 'minEpsilon' ? '' : '+') + val + suffix;
        };
    };

    bindSlider('slider-block', 'val-reward-block', 'block', ' XP');
    bindSlider('slider-three', 'val-reward-three', 'lineOfThree', ' XP');
    bindSlider('slider-center', 'val-reward-center', 'center', ' XP');
    bindSlider('slider-epsilon', 'val-epsilon', 'minEpsilon', ' %', 0.01);
}

// ==========================================
// 2. RENDU DU PLATEAU, JAUGES & POPUPS
// ==========================================

function updateGhostPreview(hintCol = null) {
    for (let r = 0; r < engine.ROWS; r++) {
        for (let c = 0; c < engine.COLS; c++) {
            const cell = document.getElementById(`cell-${r}-${c}`);
            if (cell) cell.classList.remove('ghost-preview', 'hint-preview');
        }
    }

    const targetCol = hintCol !== null ? hintCol : hoveredCol;
    if (targetCol === null || isTraining || isArena || isProcessingMove) return;

    for (let r = engine.ROWS - 1; r >= 0; r--) {
        if (engine.board[r][targetCol] === 0) {
            const cell = document.getElementById(`cell-${r}-${targetCol}`);
            if (cell) cell.classList.add(hintCol !== null ? 'hint-preview' : 'ghost-preview');
            break;
        }
    }
}

function renderBoard(lastMove = null, winningCells = null) {
    const winSet = new Set(winningCells ? winningCells.map(([r, c]) => `${r}-${c}`) : []);
    const animateDrop = currentSpeed === '1' || (!isTraining && !isArena);

    for (let r = 0; r < engine.ROWS; r++) {
        for (let c = 0; c < engine.COLS; c++) {
            const cell = document.getElementById(`cell-${r}-${c}`);
            if (!cell) continue;
            const val = engine.board[r][c];
            cell.className = 'cell' + (val === 1 ? ' player' : val === 2 ? ' ai' : '');

            if (animateDrop && lastMove && lastMove.r === r && lastMove.c === c) {
                void cell.offsetWidth;
                cell.classList.add('drop-anim');
            }
            if (winSet.has(`${r}-${c}`)) cell.classList.add('win-cell');
        }
    }
}

function updateBrainBars(scores, chosenCol, exploratory, aiLabel) {
    const tag = document.getElementById('brain-mode-label');
    if (tag) {
        tag.innerText = exploratory
            ? `🎲 ${aiLabel} explore (Curiosité)`
            : `🎯 ${aiLabel} → Colonne ${chosenCol + 1}`;
    }

    const validScores = scores.filter(s => Number.isFinite(s));
    const minS = validScores.length ? Math.min(...validScores) : -10;
    const maxS = validScores.length ? Math.max(...validScores) : 10;
    const spread = Math.max(1, maxS - minS);

    for (let c = 0; c < engine.COLS; c++) {
        const colEl = document.getElementById(`q-col-${c}`);
        const scoreEl = document.getElementById(`q-score-${c}`);
        const fillEl = document.getElementById(`q-fill-${c}`);
        if (!colEl || !scoreEl || !fillEl) continue;

        colEl.classList.toggle('best-choice', c === chosenCol);
        const s = scores[c];

        if (!Number.isFinite(s)) {
            scoreEl.innerText = 'Plein';
            fillEl.style.height = '8%';
            fillEl.style.backgroundColor = '#2f3542';
            continue;
        }

        scoreEl.innerText = (s >= 0 ? '+' : '') + s.toFixed(1);
        const ratio = Math.max(0.12, Math.min(1, (s - minS) / spread));
        fillEl.style.height = `${Math.round(ratio * 100)}%`;
        fillEl.style.backgroundColor = c === chosenCol ? (exploratory ? '#ffa502' : '#2ed573') : (s < 0 ? '#ff4757' : '#70a1ff');
    }

    visualizer.drawNeuralNetwork(brain.lastInspection);
}

function showPopup(text, color = '#2ed573') {
    if ((isTraining || isArena) && currentSpeed !== '1' && currentSpeed !== '5') return;
    const layer = document.getElementById('xp-layer');
    if (!layer) return;

    const popup = document.createElement('div');
    popup.className = 'xp-popup';
    popup.style.color = color;
    popup.innerText = text;
    layer.appendChild(popup);
    setTimeout(() => popup.remove(), 1200);
}

function updateTelemetry() {
    document.getElementById('stat-age-a').innerText = `${brain.agents['A'].games} parties`;
    document.getElementById('stat-age-b').innerText = `${brain.agents['B'].games} parties`;
    const mem = brain.agents['A'].memory.length + brain.agents['B'].memory.length;
    document.getElementById('stat-mem').innerText = `${(mem / 1000).toFixed(1)}k / 100k`;
    document.getElementById('stat-winrate').innerText = `${arenaStats.aWins} / ${arenaStats.draws} / ${arenaStats.bWins}`;
}

function setStatus(text, cls = '') {
    const el = document.getElementById('status');
    el.className = 'status-text ' + cls;
    el.innerText = text;
}

// ==========================================
// 3. JEU HUMAIN, CONSEIL IA & ANNULATION
// ==========================================

async function handleHumanMove(col) {
    if (isTraining || isArena || isProcessingMove || engine.board[0][col] !== 0) return;
    isProcessingMove = true;

    try {
        const humanCoords = engine.dropToken(col, 1);
        if (!humanCoords) return;

        const winCoords = engine.checkWinner(1);
        const humanWon = winCoords !== null;
        const isDraw = !humanWon && engine.isFull();

        renderBoard(humanCoords, winCoords);

        if (humanWon || isDraw) {
            if (humanWon) {
                setStatus("🎉 Bravo ! Tu as battu l'IA-A ! Punition mémorisée.", "status-win-human");
                showPopup("💥 Punition IA : -100 XP (x50 en mémoire)", "#ff4757");
                if (lastAITurnContext) {
                    brain.saveMemory('A', lastAITurnContext.state, lastAITurnContext.action, engine.rewardConfig.lose, lastAITurnContext.nextState, true, 25);
                    await brain.trainBatch('A', 128);
                    await brain.trainBatch('A', 128);
                    brain.agents['A'].games++;
                    await brain.checkpoint('A');
                    updateTelemetry();
                }
            } else {
                setStatus("🤝 Match nul !");
            }
            lastAITurnContext = null;
            return;
        }

        // Réflexion de l'IA-A
        setStatus("🤔 L'IA-A analyse le plateau...", "status-thinking");
        const { col: aiCol, scores, normState: stateBefore } = brain.selectAction(engine.board, 'A', 2, 0, true);
        updateBrainBars(scores, aiCol, false, 'IA-A');

        await new Promise(r => setTimeout(r, 420));

        if (aiCol !== undefined) {
            const aiCoords = engine.dropToken(aiCol, 2);
            const aiWinCoords = engine.checkWinner(2);
            const aiWon = aiWinCoords !== null;
            const aiDraw = !aiWon && engine.isFull();
            const done = aiWon || aiDraw;

            renderBoard(aiCoords, aiWinCoords);

            const { reward, popup } = engine.evaluateMoveReward(aiCol, 2, aiWon, aiDraw);
            if (popup) showPopup(popup.text, popup.color);

            const stateAfter = brain.getNormalizedState(engine.board, 2);
            lastAITurnContext = { state: stateBefore, action: aiCol, nextState: stateAfter };

            brain.saveMemory('A', stateBefore, aiCol, reward, stateAfter, done, 25);
            await brain.trainBatch('A', 128);
            updateTelemetry();

            if (aiWon) setStatus("🤖 L'IA-A remporte la manche !", "status-win-ai");
            else if (aiDraw) setStatus("🤝 Match nul !");
            else setStatus("À ton tour ! Clique sur une colonne.");

            if (done) {
                brain.agents['A'].games++;
                await brain.checkpoint('A');
                lastAITurnContext = null;
            }
        }
    } finally {
        isProcessingMove = false;
        updateGhostPreview();
    }
}

// ==========================================
// 4. ENTRAÎNEMENT SOLO & ARÈNE MULTI-VITESSES
// ==========================================

async function runTrainingLoop(mode = 'A') {
    if (isTraining || isArena || isProcessingMove) return;
    isArena = (mode === 'ARENA');
    isTraining = !isArena;
    stopRequested = false;

    let sessionGames = 0;

    while (!stopRequested) {
        sessionGames++;
        engine.reset();
        let turn = Math.random() < 0.5 ? 1 : 2;
        let history = [];
        let gameRewardSum = 0;
        let gameDone = false;

        while (!gameDone && !stopRequested) {
            const activeAI = isArena ? (turn === 1 ? 'A' : 'B') : mode;
            const isVisual = currentSpeed !== 'MAX';
            const baseEps = engine.rewardConfig.minEpsilon;
            const eps = isArena ? Math.max(0.04, baseEps / 2) : Math.max(baseEps, 0.8 - (brain.agents[activeAI].games / 8000));

            const { col, scores, exploratory, normState: stateBefore } = brain.selectAction(
                engine.board, activeAI, turn, eps, isVisual
            );
            if (col === undefined) break;

            const coords = engine.dropToken(col, turn);
            if (!coords) break;

            const winCoords = engine.checkWinner(turn);
            const win = winCoords !== null;
            const draw = !win && engine.isFull();
            gameDone = win || draw;

            const { reward, popup } = engine.evaluateMoveReward(col, turn, win, draw);
            gameRewardSum += reward;
            const stateAfter = brain.getNormalizedState(engine.board, turn);

            brain.saveMemory(activeAI, stateBefore, col, reward, stateAfter, gameDone, 1);
            history.push({ aiKey: activeAI, stateBefore, col, stateAfter });

            if (win && history.length >= 2) {
                const loser = history[history.length - 2];
                brain.saveMemory(loser.aiKey, loser.stateBefore, loser.col, engine.rewardConfig.lose, loser.stateAfter, true, isArena ? 5 : 2);
            }

            // Affichage synchronisé selon la vitesse sélectionnée (x1, x5, x10, x20)
            if (isVisual) {
                updateBrainBars(scores, col, exploratory, `IA-${activeAI}`);
                renderBoard(coords, winCoords);
                if (popup) showPopup(`IA-${activeAI}: ${popup.text}`, popup.color);

                const delay = SPEED_DELAYS[currentSpeed] || 20;
                await new Promise(r => setTimeout(r, gameDone ? delay * 2.2 : delay));
            }

            if (gameDone && isArena) {
                if (win) {
                    if (turn === 1) arenaStats.aWins++;
                    else arenaStats.bWins++;
                } else {
                    arenaStats.draws++;
                }
            }

            turn = turn === 1 ? 2 : 1;
        }

        if (isArena) {
            brain.agents['A'].games++;
            brain.agents['B'].games++;
            await brain.trainBatch('A', 128);
            await brain.trainBatch('B', 128);
        } else {
            brain.agents[mode].games++;
            await brain.trainBatch(mode, 128);
        }

        visualizer.recordGameReward(gameRewardSum);

        if (currentSpeed !== 'MAX' || sessionGames % 5 === 0) {
            setStatus(
                isArena
                    ? `⚔️ Arène (${currentSpeed === 'MAX' ? 'MAX' : 'x' + currentSpeed}) : ${sessionGames} duels joués`
                    : `💪 Entraînement IA-${mode} (${currentSpeed === 'MAX' ? 'MAX' : 'x' + currentSpeed}) : +${sessionGames} parties`
            );
            if (currentSpeed === 'MAX') renderBoard();
            updateTelemetry();
            await tf.nextFrame();
        }

        if (sessionGames % 50 === 0) {
            if (isArena) {
                await brain.checkpoint('A');
                await brain.checkpoint('B');
            } else {
                await brain.checkpoint(mode);
            }
        }
    }

    if (isArena) {
        await brain.checkpoint('A');
        await brain.checkpoint('B');
    } else {
        await brain.checkpoint(mode);
    }

    isTraining = false;
    isArena = false;
    updateTelemetry();
    engine.reset();
    renderBoard();
    setStatus(`Session terminée (${sessionGames} parties). À toi de jouer !`);
}

// ==========================================
// 5. ÉVÉNEMENTS BOUTONS
// ==========================================

function bindButtons() {
    const toggleLoop = (mode, btnId, defaultLabel) => {
        const btn = document.getElementById(btnId);
        if ((isTraining || isArena) && !stopRequested) {
            stopRequested = true;
            btn.innerText = "Arrêt en cours...";
        } else if (!isTraining && !isArena) {
            btn.innerText = "🛑 Stopper";
            runTrainingLoop(mode).then(() => { btn.innerText = defaultLabel; });
        }
    };

    document.getElementById('btn-train-a').onclick = () => toggleLoop('A', 'btn-train-a', "💪 Entraîner IA-A");
    document.getElementById('btn-train-b').onclick = () => toggleLoop('B', 'btn-train-b', "💪 Entraîner IA-B");
    document.getElementById('btn-arena').onclick = () => toggleLoop('ARENA', 'btn-arena', "⚔️ Arène (A vs B)");

    document.getElementById('btn-reset').onclick = () => {
        stopRequested = true;
        isTraining = false;
        isArena = false;
        lastAITurnContext = null;
        engine.reset();
        renderBoard();
        setStatus("Nouvelle partie ! Pose un pion rouge.");
    };

    document.getElementById('btn-undo').onclick = () => {
        if (isTraining || isArena || isProcessingMove) return;
        if (engine.undoLastExchange()) {
            renderBoard();
            showPopup("⏪ Dernier tour annulé ! Teste une autre colonne.", "#70a1ff");
            setStatus("Coup annulé. Essaie une autre stratégie !");
        } else {
            showPopup("Aucun coup à annuler", "#ff6b81");
        }
    };

    document.getElementById('btn-hint').onclick = () => {
        if (isTraining || isArena || isProcessingMove) return;
        const { col, scores } = brain.selectAction(engine.board, 'A', 1, 0, true);
        if (col !== undefined) {
            updateBrainBars(scores, col, false, 'Conseil IA-A');
            updateGhostPreview(col);
            showPopup(`💡 L'IA-A conseille la colonne ${col + 1}`, '#2ed573');
        }
    };

    document.getElementById('btn-wipe').onclick = async () => {
        if (isTraining || isArena || isProcessingMove) return;
        if (!confirm("Remettre les cerveaux IA-A et IA-B au Niveau 0 (Amnésie totale) ?")) return;
        await brain.wipeAll();
        arenaStats = { aWins: 0, bWins: 0, draws: 0 };
        visualizer.rewardHistory = [];
        engine.reset();
        renderBoard();
        updateTelemetry();
        visualizer.drawNeuralNetwork(null);
        visualizer.drawLearningCurve();
        showPopup("🧹 Cerveaux réinitialisés au Niveau 0 !", "#ff6b81");
        setStatus("Cerveaux vierges ! À toi de jouer.");
    };
}

// Démarrage de l'application
(async function bootstrap() {
    initDOM();
    bindButtons();
    await brain.init();
    document.getElementById('ia-status').innerText = "Double DQN Actif • Sonde Neuronale Temps Réel";
    updateTelemetry();
    renderBoard();
    setStatus("À toi de jouer ! Clique sur une colonne.");
})();