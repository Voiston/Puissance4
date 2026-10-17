const engine = new Connect4Engine(6, 7);
const brain = new NeuralAgentManager(6, 7);
const visualizer = new NeuralVisualizer('nn-canvas', 'chart-canvas');
const sfx = new SoundFX();

let isTraining = false;
let isArena = false;
let stopRequested = false;
let isProcessingMove = false;
let currentSpeed = '1';
let gameMode = 'HUMAN_FIRST';
let thermalVision = true;
let humanTurnPlayer = 1;
let hoveredCol = null;
let lastAITurnContext = null;
let lastKnownScores = [0, 0, 0, 0, 0, 0, 0];
let arenaStats = { aWins: 0, bWins: 0, draws: 0 };

const SPEED_DELAYS = { '1': 550, '5': 110, '10': 45, '20': 15, 'MAX': 0 };

// Visages Chibi (Kaomoji) de Mochi-IA selon son humeur
const MASCOT_FACES = {
    IDLE: '( ◕ ㅅ ◕ )',
    THINK: '( ◔ ㅅ ◔ )💭',
    HAPPY: '( ≧ ◡ ≦ )✨',
    PROUD: '( •̀ ᴗ •́ )🛡️',
    SAD: '( ╥ ﹏ ╥ )💦',
    TRAIN: '( ง •̀_•́)ง🔥',
    COACH: '( ◕ ‿ ◕ )💡'
};

function setMascotMood(moodKey) {
    const el = document.getElementById('mascot-avatar');
    if (!el) return;
    el.innerText = MASCOT_FACES[moodKey] || MASCOT_FACES.IDLE;
    el.classList.remove('bounce');
    void el.offsetWidth;
    el.classList.add('bounce');
    setTimeout(() => el.classList.remove('bounce'), 300);
}

function initDOM() {
    const gridEl = document.getElementById('board');
    gridEl.innerHTML = '';
    for (let r = 0; r < engine.ROWS; r++) {
        for (let c = 0; c < engine.COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell';
            div.id = `cell-${r}-${c}`;
            div.onclick = () => { if (!isArena && !isTraining) handleHumanMove(c); };
            div.onmouseenter = () => { hoveredCol = c; updateGhostAndThermal(); };
            div.onmouseleave = () => { hoveredCol = null; updateGhostAndThermal(); };
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

    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSpeed = btn.dataset.speed;
        };
    });

    const bindSlider = (id, labelId, key, suffix = '', scale = 1) => {
        const el = document.getElementById(id);
        const lbl = document.getElementById(labelId);
        if (!el || !lbl) return;
        el.oninput = () => {
            const val = parseFloat(el.value);
            engine.rewardConfig[key] = val * scale;
            lbl.innerText = (key === 'minEpsilon' ? '' : '+') + val + suffix;
        };
    };

    bindSlider('slider-block', 'val-reward-block', 'block', '');
    bindSlider('slider-three', 'val-reward-three', 'lineOfThree', '');
    bindSlider('slider-center', 'val-reward-center', 'center', '');
    bindSlider('slider-epsilon', 'val-epsilon', 'minEpsilon', ' %', 0.01);
}

function updateGhostAndThermal(hintCol = null) {
    for (let r = 0; r < engine.ROWS; r++) {
        for (let c = 0; c < engine.COLS; c++) {
            const cell = document.getElementById(`cell-${r}-${c}`);
            if (!cell) continue;
            cell.classList.remove('ghost-preview', 'hint-preview');
            if (engine.board[r][c] === 0) {
                cell.style.boxShadow = '';
                cell.innerHTML = '';
            }
        }
    }

    if (isTraining || isArena) return;

    if (thermalVision && lastKnownScores.length === 7) {
        for (let c = 0; c < engine.COLS; c++) {
            const r = engine.getLowestEmptyRow(c);
            const score = lastKnownScores[c];
            if (r !== -1 && Number.isFinite(score)) {
                const cell = document.getElementById(`cell-${r}-${c}`);
                if (cell) {
                    const color = score >= 1.5 ? '#2d936c' : (score < -1.5 ? '#d64568' : '#5c80bc');
                    const bgGlow = score >= 1.5 ? '#6bcf9755' : (score < -1.5 ? '#ff85a155' : '#a3c4f355');
                    cell.style.boxShadow = `inset 0 0 12px ${bgGlow}`;
                    cell.innerHTML = `<span class="thermal-badge" style="color:${color}">${score >= 0 ? '+' : ''}${score.toFixed(1)}</span>`;
                }
            }
        }
    }

    const targetCol = hintCol !== null ? hintCol : hoveredCol;
    if (targetCol === null || isProcessingMove) return;

    const r = engine.getLowestEmptyRow(targetCol);
    if (r !== -1) {
        const cell = document.getElementById(`cell-${r}-${targetCol}`);
        if (cell) cell.classList.add(hintCol !== null ? 'hint-preview' : 'ghost-preview');
    }
}

function updateEvalBar(scores, perspectivePlayer = 2) {
    const valid = scores.filter(s => Number.isFinite(s));
    if (!valid.length) return;
    const bestQ = Math.max(...valid);

    const yellowAdvantage = perspectivePlayer === 2 ? bestQ : -bestQ;
    const pctYellow = Math.round(50 + Math.tanh(yellowAdvantage / 8) * 46);

    const fillEl = document.getElementById('eval-bar-fill');
    const txtEl = document.getElementById('eval-bar-text');
    if (fillEl) fillEl.style.height = `${pctYellow}%`;
    if (txtEl) txtEl.innerText = (yellowAdvantage >= 0 ? '+' : '') + yellowAdvantage.toFixed(1);
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
            if (val !== 0) {
                cell.style.boxShadow = '';
                cell.innerHTML = '';
            }

            if (animateDrop && lastMove && lastMove.r === r && lastMove.c === c) {
                void cell.offsetWidth;
                cell.classList.add('drop-anim');
            }
            if (winSet.has(`${r}-${c}`)) cell.classList.add('win-cell');
        }
    }
    updateGhostAndThermal();
}

function updateBrainBars(scores, chosenCol, exploratory, aiLabel, perspectivePlayer = 2) {
    lastKnownScores = scores;
    const tag = document.getElementById('brain-mode-label');
    if (tag) {
        tag.innerText = exploratory ? `🦋 ${aiLabel} est curieuse` : `🌸 ${aiLabel} aime la Col. ${chosenCol + 1}`;
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
            fillEl.style.height = '10%';
            fillEl.style.backgroundColor = '#d8d0c5';
            continue;
        }

        scoreEl.innerText = (s >= 0 ? '+' : '') + s.toFixed(1);
        const ratio = Math.max(0.14, Math.min(1, (s - minS) / spread));
        fillEl.style.height = `${Math.round(ratio * 100)}%`;
        fillEl.style.backgroundColor = c === chosenCol ? (exploratory ? '#ffca5c' : '#6bcf97') : (s < 0 ? '#ff85a1' : '#8ecae6');
    }

    updateEvalBar(scores, perspectivePlayer);
    visualizer.drawNeuralNetwork(brain.lastInspection);
    updateGhostAndThermal();
}

function showPopup(text, color = '#3b8c62') {
    if ((isTraining || isArena) && currentSpeed !== '1' && currentSpeed !== '5') return;
    const layer = document.getElementById('xp-layer');
    if (!layer) return;
    const popup = document.createElement('div');
    popup.className = 'xp-popup';
    popup.style.color = color;
    popup.innerText = text;
    layer.appendChild(popup);
    setTimeout(() => popup.remove(), 1250);
}

function updateTelemetry() {
    const gamesA = brain.agents['A'].games;
    const gamesB = brain.agents['B'].games;
    document.getElementById('stat-age-a').innerText = `${gamesA} parties`;
    document.getElementById('stat-age-b').innerText = `${gamesB} parties`;
    document.getElementById('rank-a').innerText = brain.getRankTitle(gamesA);
    document.getElementById('rank-b').innerText = brain.getRankTitle(gamesB);

    const mem = brain.agents['A'].memory.length + brain.agents['B'].memory.length;
    document.getElementById('stat-mem').innerText = `${(mem / 1000).toFixed(1)}k / 50k`;
    document.getElementById('stat-winrate').innerText = `${arenaStats.aWins} / ${arenaStats.draws} / ${arenaStats.bWins}`;
    document.getElementById('stat-curriculum').innerText = gamesA < 500 ? "🦉 Hibou Actif" : "🦋 Autonome";
}

function setStatus(text, cls = '') {
    const el = document.getElementById('status');
    el.className = 'status-text ' + cls;
    el.innerText = text;
}

async function triggerAIResponse() {
    setMascotMood('THINK');
    setStatus("Hmm... Mochi réfléchit à son coup...", "status-thinking");

    let aiCol;
    let scores = [0, 0, 0, 0, 0, 0, 0];
    let stateBefore = brain.getNormalizedState(engine.board, 2);

    if (gameMode === 'VS_TEACHER') {
        aiCol = engine.getTeacherMove(2, 'GUARDIAN');
        const res = brain.selectAction(engine.board, 'A', 2, 0, true);
        scores = res.scores;
        updateBrainBars(scores, aiCol, false, 'Maître Hibou', 2);
    } else {
        const res = brain.selectAction(engine.board, 'A', 2, 0, true);
        aiCol = res.col;
        scores = res.scores;
        stateBefore = res.normState;
        updateBrainBars(scores, aiCol, false, 'Mochi-A', 2);
    }

    await new Promise(r => setTimeout(r, 420));

    if (aiCol !== undefined) {
        const aiCoords = engine.dropToken(aiCol, 2);
        sfx.drop(false);
        const aiWinCoords = engine.checkWinner(2);
        const aiWon = aiWinCoords !== null;
        const aiDraw = !aiWon && engine.isFull();
        const done = aiWon || aiDraw;

        renderBoard(aiCoords, aiWinCoords);
        const { reward, popup } = engine.evaluateMoveReward(aiCol, 2, aiWon, aiDraw);
        if (popup) {
            showPopup(popup.text, '#2d7a50');
            if (reward >= 5) {
                sfx.reward();
                setMascotMood('PROUD');
            }
        }

        const stateAfter = brain.getNormalizedState(engine.board, 2);
        lastAITurnContext = { state: stateBefore, action: aiCol, nextState: stateAfter };

        brain.saveMemory('A', stateBefore, aiCol, reward, stateAfter, done, 25);
        await brain.trainBatch('A', 128);

        const nextHumanEval = brain.selectAction(engine.board, 'A', 1, 0, false);
        lastKnownScores = nextHumanEval.scores;
        updateGhostAndThermal();
        updateTelemetry();

        if (aiWon) {
            sfx.win();
            setMascotMood('HAPPY');
            setStatus("Youpi ! Mochi a aligné 4 pions !", "status-win-ai");
        } else if (aiDraw) {
            setMascotMood('IDLE');
            setStatus("🤝 Égalité parfaite ! Belle partie !");
        } else {
            if (reward < 5) setMascotMood('IDLE');
            setStatus("À toi de jouer ! Clique sur une colonne.");
        }

        if (done) {
            brain.agents['A'].games++;
            await brain.checkpoint('A');
            lastAITurnContext = null;
        }
    }
}

async function handleHumanMove(col) {
    if (isTraining || isArena || isProcessingMove || engine.board[0][col] !== 0) return;
    isProcessingMove = true;

    try {
        if (gameMode === 'COACH_2P') {
            const coords = engine.dropToken(col, humanTurnPlayer);
            if (!coords) return;
            sfx.drop(humanTurnPlayer === 1);

            const winCoords = engine.checkWinner(humanTurnPlayer);
            const won = winCoords !== null;
            const draw = !won && engine.isFull();
            renderBoard(coords, winCoords);

            if (won) {
                sfx.win();
                setMascotMood('HAPPY');
                setStatus(`🎉 Bravo ! Le pion ${humanTurnPlayer === 1 ? '🍓 Fraise' : '🍯 Miel'} gagne !`, humanTurnPlayer === 1 ? 'status-win-human' : 'status-win-ai');
                return;
            }
            if (draw) {
                setStatus("🤝 Match nul !");
                return;
            }

            humanTurnPlayer = humanTurnPlayer === 1 ? 2 : 1;
            const coach = brain.selectAction(engine.board, 'A', humanTurnPlayer, 0, true);
            setMascotMood('COACH');
            updateBrainBars(coach.scores, coach.col, false, `Coach J${humanTurnPlayer}`, humanTurnPlayer);
            setStatus(`Tour de ${humanTurnPlayer === 1 ? '🍓 Fraise' : '🍯 Miel'} (Mochi suggère la Col. ${coach.col + 1})`);
            return;
        }

        const humanCoords = engine.dropToken(col, 1);
        if (!humanCoords) return;
        sfx.drop(true);

        const winCoords = engine.checkWinner(1);
        const humanWon = winCoords !== null;
        const isDraw = !humanWon && engine.isFull();
        renderBoard(humanCoords, winCoords);

        if (humanWon || isDraw) {
            if (humanWon) {
                sfx.punish();
                setMascotMood('SAD');
                setStatus("Oups ! Tu as gagné ! Mochi note son erreur pour la prochaine fois.", "status-win-human");
                showPopup("💦 Leçon apprise par Mochi (-10)", "#d64568");
                if (lastAITurnContext) {
                    brain.saveMemory('A', lastAITurnContext.state, lastAITurnContext.action, -10, lastAITurnContext.nextState, true, 25);
                    await brain.trainBatch('A', 128);
                    brain.agents['A'].games++;
                    await brain.checkpoint('A');
                    updateTelemetry();
                }
            } else {
                setMascotMood('IDLE');
                setStatus("🤝 Match nul !");
            }
            lastAITurnContext = null;
            return;
        }

        await triggerAIResponse();
    } finally {
        isProcessingMove = false;
        updateGhostAndThermal();
    }
}

async function runTrainingLoop(mode = 'A') {
    if (isTraining || isArena || isProcessingMove) return;
    isArena = (mode === 'ARENA');
    isTraining = !isArena;
    stopRequested = false;

    setMascotMood('TRAIN');
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
            const eps = isArena ? 0.05 : Math.max(engine.rewardConfig.minEpsilon, 0.8 - (brain.agents[activeAI].games / 6000));

            let col;
            let scores = Array(7).fill(0);
            let exploratory = false;
            let stateBefore;

            if (!isArena && brain.agents[mode].games < 500 && turn === 2) {
                col = engine.getTeacherMove(2, 'GUARDIAN');
                stateBefore = brain.getNormalizedState(engine.board, 2);
            } else {
                const res = brain.selectAction(engine.board, activeAI, turn, eps, isVisual);
                col = res.col;
                scores = res.scores;
                exploratory = res.exploratory;
                stateBefore = res.normState;
            }

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

            history.push({ aiKey: activeAI, stateBefore, col, stateAfter, immediateReward: reward });

            if (isVisual) {
                if (currentSpeed === '1') sfx.drop(turn === 1);
                updateBrainBars(scores, col, exploratory, `Mochi-${activeAI}`, turn);
                renderBoard(coords, winCoords);
                if (popup) showPopup(`Mochi-${activeAI}: ${popup.text}`, '#2d7a50');
                const delay = SPEED_DELAYS[currentSpeed] || 20;
                await new Promise(r => setTimeout(r, gameDone ? delay * 2 : delay));
            }

            if (gameDone && isArena) {
                if (win) { if (turn === 1) arenaStats.aWins++; else arenaStats.bWins++; }
                else { arenaStats.draws++; }
            }

            turn = turn === 1 ? 2 : 1;
        }

        if (history.length > 0) {
            let finalOutcomeReward = history[history.length - 1].immediateReward > 5 ? 10 : (gameDone && engine.checkWinner(1) ? -10 : 0.5);
            let propagated = finalOutcomeReward;
            const DECAY = 0.85;

            for (let i = history.length - 1; i >= 0; i--) {
                const h = history[i];
                const totalR = h.immediateReward + propagated;
                brain.saveMemory(h.aiKey, h.stateBefore, h.col, totalR, h.stateAfter, i >= history.length - 2, 1);
                propagated = -propagated * DECAY;
            }
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
                    ? `🎡 Match amical (${currentSpeed === 'MAX' ? 'MAX' : 'x' + currentSpeed}) : ${sessionGames} parties`
                    : `🍯 Mochi-${mode} s'entraîne (${currentSpeed === 'MAX' ? 'MAX' : 'x' + currentSpeed}) : +${sessionGames} parties`
            );
            if (currentSpeed === 'MAX') renderBoard();
            updateTelemetry();
            await tf.nextFrame();
        }

        if (sessionGames % 50 === 0) {
            if (isArena) { await brain.checkpoint('A'); await brain.checkpoint('B'); }
            else { await brain.checkpoint(mode); }
        }
    }

    if (isArena) { await brain.checkpoint('A'); await brain.checkpoint('B'); }
    else { await brain.checkpoint(mode); }

    isTraining = false;
    isArena = false;
    updateTelemetry();
    startNewHumanGame();
}

async function startNewHumanGame() {
    stopRequested = true;
    isTraining = false;
    isArena = false;
    lastAITurnContext = null;
    humanTurnPlayer = 1;
    engine.reset();

    setMascotMood('IDLE');
    const initEval = brain.selectAction(engine.board, 'A', 1, 0, true);
    lastKnownScores = initEval.scores;
    renderBoard();

    if (gameMode === 'AI_FIRST') {
        isProcessingMove = true;
        await triggerAIResponse();
        isProcessingMove = false;
    } else if (gameMode === 'COACH_2P') {
        setMascotMood('COACH');
        updateBrainBars(initEval.scores, initEval.col, false, 'Coach J1', 1);
        setStatus("👯 Mode 2 Joueurs : Au tour de 🍓 Fraise !");
    } else {
        setStatus("Coucou ! Pose un pion 🍓 Fraise pour commencer.");
    }
}

function bindButtons() {
    const toggleLoop = (mode, btnId, defaultLabel) => {
        const btn = document.getElementById(btnId);
        if ((isTraining || isArena) && !stopRequested) {
            stopRequested = true;
            btn.innerText = "Pause...";
        } else if (!isTraining && !isArena) {
            btn.innerText = "🛑 Pause";
            runTrainingLoop(mode).then(() => { btn.innerText = defaultLabel; });
        }
    };

    document.getElementById('btn-train-a').onclick = () => toggleLoop('A', 'btn-train-a', "🍯 Entraîner A");
    document.getElementById('btn-train-b').onclick = () => toggleLoop('B', 'btn-train-b', "🫐 Entraîner B");
    document.getElementById('btn-arena').onclick = () => toggleLoop('ARENA', 'btn-arena', "🎡 Match A vs B");

    document.getElementById('btn-reset').onclick = () => startNewHumanGame();

    document.getElementById('game-mode').onchange = (e) => {
        gameMode = e.target.value;
        startNewHumanGame();
    };

    document.getElementById('btn-thermal').onclick = (e) => {
        thermalVision = !thermalVision;
        e.currentTarget.classList.toggle('active', thermalVision);
        e.currentTarget.innerText = `🌸 Bulles d'Intuition : ${thermalVision ? 'ON' : 'OFF'}`;
        updateGhostAndThermal();
    };

    document.getElementById('btn-sound').onclick = (e) => {
        sfx.enabled = !sfx.enabled;
        e.currentTarget.innerText = sfx.enabled ? '🔊' : '🔇';
    };

    document.getElementById('btn-undo').onclick = () => {
        if (isTraining || isArena || isProcessingMove) return;
        const steps = gameMode === 'COACH_2P' ? 1 : 2;
        if (engine.undoLastExchange(steps)) {
            if (gameMode === 'COACH_2P') humanTurnPlayer = humanTurnPlayer === 1 ? 2 : 1;
            renderBoard();
            setMascotMood('IDLE');
            showPopup("⏪ Tour annulé", "#548cc4");
        } else {
            showPopup("Rien à annuler", "#d64568");
        }
    };

    document.getElementById('btn-hint').onclick = () => {
        if (isTraining || isArena || isProcessingMove) return;
        const p = gameMode === 'COACH_2P' ? humanTurnPlayer : 1;
        const { col, scores } = brain.selectAction(engine.board, 'A', p, 0, true);
        if (col !== undefined) {
            setMascotMood('COACH');
            updateBrainBars(scores, col, false, 'Indice Mochi', p);
            updateGhostAndThermal(col);
            showPopup(`💡 Mochi conseille la Colonne ${col + 1}`, '#2d7a50');
        }
    };

    document.getElementById('btn-clone-a-to-b').onclick = async () => {
        if (isTraining || isArena) return;
        await brain.cloneAgent('A', 'B');
        updateTelemetry();
        setMascotMood('HAPPY');
        showPopup("👯 Mochi-A copiée dans Mochi-B !", "#2d7a50");
    };

    document.getElementById('btn-export-a').onclick = () => brain.exportModel('A');

    document.getElementById('btn-wipe').onclick = async () => {
        if (isTraining || isArena || isProcessingMove) return;
        if (!confirm("Faire oublier tout ce que Mochi-A et Mochi-B ont appris (Niveau 0) ?")) return;
        await brain.wipeAll();
        arenaStats = { aWins: 0, bWins: 0, draws: 0 };
        visualizer.rewardHistory = [];
        updateTelemetry();
        visualizer.drawNeuralNetwork(null);
        visualizer.drawLearningCurve();
        showPopup("🧹 Mémoire toute propre !", "#d64568");
        startNewHumanGame();
    };
}

(async function bootstrap() {
    initDOM();
    bindButtons();
    await brain.init();
    document.getElementById('ia-status').innerText = "Édition Chibi Cozy • Neurones Prêts & Boîte à Musique Active";
    updateTelemetry();
    startNewHumanGame();
})();