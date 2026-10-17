const ROWS = 6, COLS = 7;
let board = [];
let isTraining = false;
let isArena = false;
let stopTrainingRequested = false;
let stopArenaRequested = false;
let isProcessingMove = false; // Verrou anti-autoclicker
let isSpectatorMode = false;  // Mode lent pédagogique vs Mode Turbo
let lastAITurnContext = null; // Contexte pour punir l'IA si l'humain gagne
let hoveredCol = null;

// Paramètres DQN
const GAMMA = 0.99;
const MEMORY_SIZE = 50000;

// Nos deux gladiateurs (Passage en V3 pour la vue normalisée -1 / 0 / 1)
const AIs = {
    'A': { model: null, target: null, memory: [], games: 0, storage: 'indexeddb://dqn-ia-a-v3', ageKey: 'dqn_age_a_v3' },
    'B': { model: null, target: null, memory: [], games: 0, storage: 'indexeddb://dqn-ia-b-v3', ageKey: 'dqn_age_b_v3' }
};

// ==========================================
// 1. INTERFACE VISUELLE, JAUGES & ANIMATIONS
// ==========================================

function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

function initDOMGrid() {
    const gridEl = document.getElementById('board');
    gridEl.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell';
            div.id = `cell-${r}-${c}`;
            div.onclick = () => { if (!isArena && !isTraining) handleMove(c); };
            div.onmouseenter = () => { hoveredCol = c; updateGhostPreview(); };
            div.onmouseleave = () => { hoveredCol = null; updateGhostPreview(); };
            gridEl.appendChild(div);
        }
    }

    // Initialisation des 7 jauges du cerveau
    const brainEl = document.getElementById('brain-bars');
    brainEl.innerHTML = '';
    for (let c = 0; c < COLS; c++) {
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
}

function updateGhostPreview() {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.getElementById(`cell-${r}-${c}`);
            if (cell) cell.classList.remove('ghost-preview');
        }
    }
    if (hoveredCol === null || isTraining || isArena || isProcessingMove) return;

    // Trouver la case vide la plus basse dans la colonne survolée
    for (let r = ROWS - 1; r >= 0; r--) {
        if (board[r][hoveredCol] === 0) {
            const targetCell = document.getElementById(`cell-${r}-${hoveredCol}`);
            if (targetCell) targetCell.classList.add('ghost-preview');
            break;
        }
    }
}

function renderBoard(lastMove = null, winningCells = null) {
    if (isTraining && !isSpectatorMode) return;

    const winSet = new Set(winningCells ? winningCells.map(([r, c]) => `${r}-${c}`) : []);

    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.getElementById(`cell-${r}-${c}`);
            if (!cell) continue;

            const val = board[r][c];
            cell.className = 'cell' + (val === 1 ? ' player' : val === 2 ? ' ai' : '');

            if (lastMove && lastMove.r === r && lastMove.c === c) {
                // Force le reflow pour rejouer l'animation de chute proprement
                void cell.offsetWidth;
                cell.classList.add('drop-anim');
            }

            if (winSet.has(`${r}-${c}`)) {
                cell.classList.add('win-cell');
            }
        }
    }
    updateGhostPreview();
}

// Affiche en direct les 7 évaluations du réseau de neurones au-dessus des colonnes
function updateBrainVisualizer(scores, chosenCol, isExploratory = false, aiLabel = 'IA-A') {
    const modeLabel = document.getElementById('brain-mode-label');
    if (modeLabel) {
        modeLabel.innerText = isExploratory
            ? `🎲 ${aiLabel} teste un coup au hasard (Curiosité)`
            : `🎯 ${aiLabel} choisit la colonne ${chosenCol + 1}`;
    }

    const validScores = scores.filter(s => Number.isFinite(s));
    const minScore = validScores.length ? Math.min(...validScores) : -10;
    const maxScore = validScores.length ? Math.max(...validScores) : 10;
    const spread = Math.max(1, maxScore - minScore);

    for (let c = 0; c < COLS; c++) {
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
        const ratio = Math.max(0.12, Math.min(1, (s - minScore) / spread));
        fillEl.style.height = `${Math.round(ratio * 100)}%`;

        if (c === chosenCol) {
            fillEl.style.backgroundColor = isExploratory ? '#eccc68' : '#2ed573';
        } else if (s < 0) {
            fillEl.style.backgroundColor = '#ff4757';
        } else {
            fillEl.style.backgroundColor = '#70a1ff';
        }
    }
}

// Affiche une bulle de récompense / punition façon RPG
function showFeedbackPopup(text, color = '#2ed573') {
    if ((isTraining || isArena) && !isSpectatorMode) return;
    const layer = document.getElementById('xp-layer');
    if (!layer) return;

    const popup = document.createElement('div');
    popup.className = 'xp-popup';
    popup.style.color = color;
    popup.innerText = text;
    layer.appendChild(popup);

    setTimeout(() => popup.remove(), 1400);
}

function updateTelemetryUI() {
    document.getElementById('stat-age-a').innerText = `${AIs['A'].games} parties`;
    document.getElementById('stat-age-b').innerText = `${AIs['B'].games} parties`;
    const totalMem = AIs['A'].memory.length + AIs['B'].memory.length;
    document.getElementById('stat-mem').innerText = `${totalMem} coups`;
}

function setStatus(text, statusClass = '') {
    const statusEl = document.getElementById('status');
    statusEl.className = statusClass;
    statusEl.innerText = text;
}

// ==========================================
// 2. CRÉATION ET GESTION DES DEUX CERVEAUX
// ==========================================

function compileModel(m) {
    m.compile({ optimizer: tf.train.adam(0.00025), loss: 'meanSquaredError' });
    return m;
}

function createModel() {
    const m = tf.sequential();
    m.add(tf.layers.reshape({ targetShape: [6, 7, 1], inputShape: [42] }));
    m.add(tf.layers.conv2d({ filters: 128, kernelSize: 4, activation: 'relu', padding: 'same' }));
    m.add(tf.layers.conv2d({ filters: 128, kernelSize: 4, activation: 'relu', padding: 'same' }));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 7, activation: 'linear' }));
    return compileModel(m);
}

async function initIA() {
    for (const key of ['A', 'B']) {
        AIs[key].games = parseInt(localStorage.getItem(AIs[key].ageKey) || '0', 10);
        try {
            AIs[key].model = await tf.loadLayersModel(AIs[key].storage);
            AIs[key].target = await tf.loadLayersModel(AIs[key].storage);
            compileModel(AIs[key].model);
            compileModel(AIs[key].target);
        } catch (e) {
            AIs[key].model = createModel();
            AIs[key].target = createModel();
            AIs[key].games = 0;
        }
    }

    document.getElementById('ia-status').innerText = "IA A et B prêtes (Double DQN + Symétrie + Vue Normalisée)";
    updateTelemetryUI();
    renderBoard();
    setStatus("À toi de jouer ! Pose un pion rouge.");
}

async function resetBrainsToZero() {
    if (isTraining || isArena || isProcessingMove) return;
    if (!confirm("Voulez-vous remettre les cerveaux de IA-A et IA-B au Niveau 0 (Amnésie totale) pour une démonstration ?")) return;

    for (const key of ['A', 'B']) {
        try { await tf.io.removeModel(AIs[key].storage); } catch (e) {}
        if (AIs[key].model) AIs[key].model.dispose();
        if (AIs[key].target) AIs[key].target.dispose();

        AIs[key].model = createModel();
        AIs[key].target = createModel();
        AIs[key].memory = [];
        AIs[key].games = 0;
        localStorage.setItem(AIs[key].ageKey, '0');
    }

    initBoard();
    renderBoard();
    updateTelemetryUI();
    updateBrainVisualizer([0, 0, 0, 0, 0, 0, 0], -1, false, 'IA-A');
    showFeedbackPopup("🧹 Cerveaux réinitialisés au Niveau 0 !", "#ff6b81");
    setStatus("Les IA sont redevenues naïves. À toi de jouer !");
}

// ==========================================
// 3. NORMALISATION, PRÉDICTION & SYMÉTRIE
// ==========================================

// Normalise le plateau du point de vue du joueur actif :
//  1 = Mes pions | -1 = Pions adverses | 0 = Case vide
function getNormalizedState(grid, perspectivePlayer) {
    const flat = grid.flat();
    return flat.map(v => v === 0 ? 0 : (v === perspectivePlayer ? 1 : -1));
}

function getBestMoveWithScores(grid, aiName, perspectivePlayer, epsilon = 0) {
    const validCols = [0, 1, 2, 3, 4, 5, 6].filter(c => grid[0][c] === 0);
    if (validCols.length === 0) return { col: undefined, scores: [], exploratory: false };

    const rawScores = tf.tidy(() => {
        const normState = getNormalizedState(grid, perspectivePlayer);
        const input = tf.tensor2d([normState]);
        return Array.from(AIs[aiName].model.predict(input).dataSync());
    });

    // Masquer les colonnes pleines avec -Infinity
    const maskedScores = rawScores.map((s, c) => grid[0][c] === 0 ? s : -Infinity);

    if (Math.random() < epsilon) {
        const randomCol = validCols[Math.floor(Math.random() * validCols.length)];
        return { col: randomCol, scores: maskedScores, exploratory: true };
    }

    let bestCol = validCols[0];
    for (const c of validCols) {
        if (maskedScores[c] > maskedScores[bestCol]) bestCol = c;
    }

    return { col: bestCol, scores: maskedScores, exploratory: false };
}

function getMirroredState(flatBoard) {
    const mirrored = new Array(42);
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            mirrored[r * COLS + c] = flatBoard[r * COLS + (COLS - 1 - c)];
        }
    }
    return mirrored;
}

function saveMemory(aiName, normState, action, reward, normNextState, done, clones = 1) {
    const memory = AIs[aiName].memory;
    const mirroredState = getMirroredState(normState);
    const mirroredNextState = getMirroredState(normNextState);
    const mirroredAction = (COLS - 1) - action;

    for (let i = 0; i < clones; i++) {
        memory.push({ state: normState, action, reward, nextState: normNextState, done });
        memory.push({ state: mirroredState, action: mirroredAction, reward, nextState: mirroredNextState, done });
    }

    while (memory.length > MEMORY_SIZE) memory.shift();
}

// ==========================================
// 4. ENTRAÎNEMENT DOUBLE DQN
// ==========================================

async function trainBatch(aiName, size = 128) {
    const memory = AIs[aiName].memory;
    if (memory.length < size) return;

    const batch = [];
    for (let i = 0; i < size; i++) {
        batch.push(memory[Math.floor(Math.random() * memory.length)]);
    }

    const { x, y } = tf.tidy(() => {
        const states = tf.tensor2d(batch.map(m => m.state));
        const nextStates = tf.tensor2d(batch.map(m => m.nextState));

        const currentQ = AIs[aiName].model.predict(states);
        const nextQMain = AIs[aiName].model.predict(nextStates);
        const nextQTarget = AIs[aiName].target.predict(nextStates);

        const qValues = currentQ.arraySync();
        const nextQMainValues = nextQMain.arraySync();
        const nextQTargetValues = nextQTarget.arraySync();

        batch.forEach((m, i) => {
            let target = m.reward;
            if (!m.done) {
                const bestFutureAction = nextQMainValues[i].indexOf(Math.max(...nextQMainValues[i]));
                const futureValue = nextQTargetValues[i][bestFutureAction];
                target = m.reward + GAMMA * futureValue;
            }
            qValues[i][m.action] = target;
        });

        return { x: states, y: tf.tensor2d(qValues) };
    });

    await AIs[aiName].model.fit(x, y, { epochs: 1, silent: true });
    x.dispose();
    y.dispose();
}

// ==========================================
// 5. BOUCLES D'ENTRAÎNEMENT (SOLO & ARÈNE)
// ==========================================

async function runTraining(aiName) {
    if (isTraining || isArena || isProcessingMove) return;
    isTraining = true;
    stopTrainingRequested = false;

    let sessionGames = 0;
    setStatus(`Entraînement IA-${aiName} démarré...`);

    while (!stopTrainingRequested) {
        sessionGames++;
        AIs[aiName].games++;
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2;
        let epsilon = Math.max(0.10, 0.8 - (AIs[aiName].games / 8000));
        let history = [];

        while (!stopTrainingRequested) {
            const normStateBefore = getNormalizedState(board, turn);
            const { col, scores, exploratory } = getBestMoveWithScores(board, aiName, turn, epsilon);
            if (col === undefined) break;

            const moveCoords = dropToken(board, col, turn);
            if (!moveCoords) break;

            const winCoords = checkWinner(board, turn);
            const win = winCoords !== null;
            const draw = !win && board.flat().every(v => v !== 0);
            const done = win || draw;

            const { totalReward, reason } = calculateRewardDetails(board, col, turn, win, draw);
            const normStateAfter = getNormalizedState(board, turn);

            history.push({ normStateBefore, col, totalReward, normStateAfter, done });
            saveMemory(aiName, normStateBefore, col, totalReward, normStateAfter, done, 1);

            // Punir le coup perdant de l'autre côté pour renforcer la défense
            if (win && history.length >= 2) {
                const loserMove = history[history.length - 2];
                saveMemory(aiName, loserMove.normStateBefore, loserMove.col, -100, loserMove.normStateAfter, true, 2);
            }

            if (isSpectatorMode) {
                updateBrainVisualizer(scores, col, exploratory, `IA-${aiName}`);
                renderBoard(moveCoords, winCoords);
                if (reason) showFeedbackPopup(reason.text, reason.color);
                updateTelemetryUI();
                await new Promise(r => setTimeout(r, done ? 550 : 180));
            }

            if (done) break;
            turn = (turn === 1) ? 2 : 1;
        }

        await trainBatch(aiName, 128);

        if (sessionGames % 10 === 0 || isSpectatorMode) {
            setStatus(`💪 IA-${aiName} s'entraîne : +${sessionGames} parties (Total : ${AIs[aiName].games})`);
            updateTelemetryUI();
        }

        if (sessionGames % 100 === 0) {
            AIs[aiName].target.setWeights(AIs[aiName].model.getWeights());
            await AIs[aiName].model.save(AIs[aiName].storage);
            localStorage.setItem(AIs[aiName].ageKey, AIs[aiName].games.toString());
        }

        await tf.nextFrame();
    }

    AIs[aiName].target.setWeights(AIs[aiName].model.getWeights());
    await AIs[aiName].model.save(AIs[aiName].storage);
    localStorage.setItem(AIs[aiName].ageKey, AIs[aiName].games.toString());
    isTraining = false;
    updateTelemetryUI();
    initBoard();
    renderBoard();
    setStatus(`Entraînement terminé ! IA-${aiName} a joué ${AIs[aiName].games} parties.`);
}

async function runArena() {
    if (isTraining || isArena || isProcessingMove) return;
    isArena = true;
    stopArenaRequested = false;

    let totalGames = 0;

    while (!stopArenaRequested) {
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2;
        let history = [];
        let gameDone = false;

        while (!gameDone && !stopArenaRequested) {
            const activeAI = turn === 1 ? 'A' : 'B';
            const normStateBefore = getNormalizedState(board, turn);

            const { col, scores, exploratory } = getBestMoveWithScores(board, activeAI, turn, 0.06);
            if (col === undefined) break;

            const moveCoords = dropToken(board, col, turn);
            if (!moveCoords) break;

            const winCoords = checkWinner(board, turn);
            const win = winCoords !== null;
            const draw = !win && board.flat().every(v => v !== 0);
            gameDone = win || draw;

            const { totalReward, reason } = calculateRewardDetails(board, col, turn, win, draw);
            const normStateAfter = getNormalizedState(board, turn);

            // Enregistrer les coups intermédiaires intelligents (blocages, alignements)
            saveMemory(activeAI, normStateBefore, col, totalReward, normStateAfter, gameDone, 1);

            history.push({
                aiName: activeAI,
                state: normStateBefore,
                action: col,
                nextState: normStateAfter
            });

            if (isSpectatorMode) {
                updateBrainVisualizer(scores, col, exploratory, `IA-${activeAI}`);
                renderBoard(moveCoords, winCoords);
                if (reason) showFeedbackPopup(`IA-${activeAI} : ${reason.text}`, reason.color);
                await new Promise(r => setTimeout(r, gameDone ? 650 : 200));
            }

            if (gameDone) {
                if (win) {
                    const winMove = history[history.length - 1];
                    saveMemory(winMove.aiName, winMove.state, winMove.action, 100, winMove.nextState, true, 5);
                    if (history.length >= 2) {
                        const loseMove = history[history.length - 2];
                        saveMemory(loseMove.aiName, loseMove.state, loseMove.action, -100, loseMove.nextState, true, 5);
                    }
                }
            }
            turn = (turn === 1) ? 2 : 1;
        }

        totalGames++;
        AIs['A'].games++;
        AIs['B'].games++;

        if (totalGames % 5 === 0 || isSpectatorMode) {
            setStatus(`⚔️ Arène en cours : ${totalGames} duels disputés...`);
            await trainBatch('A', 128);
            await trainBatch('B', 128);
            if (!isSpectatorMode) renderBoard();
            updateTelemetryUI();
            await tf.nextFrame();
        }

        if (totalGames % 50 === 0) {
            AIs['A'].target.setWeights(AIs['A'].model.getWeights());
            AIs['B'].target.setWeights(AIs['B'].model.getWeights());
            await AIs['A'].model.save(AIs['A'].storage);
            await AIs['B'].model.save(AIs['B'].storage);
            localStorage.setItem(AIs['A'].ageKey, AIs['A'].games.toString());
            localStorage.setItem(AIs['B'].ageKey, AIs['B'].games.toString());
        }
    }

    localStorage.setItem(AIs['A'].ageKey, AIs['A'].games.toString());
    localStorage.setItem(AIs['B'].ageKey, AIs['B'].games.toString());
    isArena = false;
    updateTelemetryUI();
    setStatus(`Arène stoppée après ${totalGames} duels.`);
}

// ==========================================
// 6. JOUER CONTRE L'HUMAIN (AVEC PÉDAGOGIE)
// ==========================================

async function handleMove(col) {
    if (isTraining || isArena || isProcessingMove || board[0][col] !== 0) return;
    isProcessingMove = true;

    try {
        // --- 1. TOUR DE L'HUMAIN (Pion 1 = Rouge) ---
        const humanCoords = dropToken(board, col, 1);
        if (!humanCoords) return;

        const humanWinCoords = checkWinner(board, 1);
        const humanWon = humanWinCoords !== null;
        const isDraw = !humanWon && board.flat().every(v => v !== 0);

        renderBoard(humanCoords, humanWinCoords);

        if (humanWon || isDraw) {
            if (humanWon) {
                setStatus("🎉 Bravo ! Tu as battu l'IA-A ! Elle analyse son erreur...", "status-win-human");
                showFeedbackPopup("💥 Punition IA : -100 XP (x50 en mémoire !)", "#ff4757");
            } else {
                setStatus("🤝 Match nul !");
            }

            if (humanWon && lastAITurnContext) {
                // Punition chirurgicale sur la décision précédente de l'IA
                saveMemory('A', lastAITurnContext.state, lastAITurnContext.action, -100, lastAITurnContext.nextState, true, 25);
                await trainBatch('A', 128);
                await trainBatch('A', 128);

                AIs['A'].games++;
                localStorage.setItem(AIs['A'].ageKey, AIs['A'].games.toString());
                AIs['A'].target.setWeights(AIs['A'].model.getWeights());
                await AIs['A'].model.save(AIs['A'].storage);
                updateTelemetryUI();
            }
            lastAITurnContext = null;
            return;
        }

        // --- 2. RÉFLEXION DE L'IA-A (Pion 2 = Jaune) ---
        setStatus("🤔 L'IA-A observe le plateau et évalue les 7 colonnes...", "status-thinking");

        const normStateBeforeAI = getNormalizedState(board, 2);
        const { col: aiCol, scores } = getBestMoveWithScores(board, 'A', 2, 0);

        // On affiche d'abord ses jauges de réflexion pendant 450ms pour que l'humain voie son raisonnement
        updateBrainVisualizer(scores, aiCol, false, 'IA-A');
        await new Promise(r => setTimeout(r, 450));

        if (aiCol !== undefined) {
            const aiCoords = dropToken(board, aiCol, 2);
            if (aiCoords) {
                const aiWinCoords = checkWinner(board, 2);
                const aiWon = aiWinCoords !== null;
                const aiDraw = !aiWon && board.flat().every(v => v !== 0);
                const done = aiWon || aiDraw;

                renderBoard(aiCoords, aiWinCoords);

                const { totalReward, reason } = calculateRewardDetails(board, aiCol, 2, aiWon, aiDraw);
                if (reason) showFeedbackPopup(reason.text, reason.color);

                const normStateAfterAI = getNormalizedState(board, 2);
                lastAITurnContext = { state: normStateBeforeAI, action: aiCol, nextState: normStateAfterAI };

                saveMemory('A', normStateBeforeAI, aiCol, totalReward, normStateAfterAI, done, 25);
                await trainBatch('A', 128);
                await trainBatch('A', 128);
                updateTelemetryUI();

                if (aiWon) {
                    setStatus("🤖 L'IA-A a gagné ! Ses connexions sont renforcées.", "status-win-ai");
                } else if (aiDraw) {
                    setStatus("🤝 Match nul !");
                } else {
                    setStatus("À ton tour ! Clique sur une colonne.");
                }

                if (done) {
                    AIs['A'].games++;
                    localStorage.setItem(AIs['A'].ageKey, AIs['A'].games.toString());
                    AIs['A'].target.setWeights(AIs['A'].model.getWeights());
                    await AIs['A'].model.save(AIs['A'].storage);
                    updateTelemetryUI();
                    lastAITurnContext = null;
                }
            }
        }
    } finally {
        isProcessingMove = false;
        updateGhostPreview();
    }
}

// ==========================================
// 7. SYSTÈME DE RÉCOMPENSES & RÈGLES DU JEU
// ==========================================

function calculateRewardDetails(board, lastAction, currentPlayer, isWin, isDraw) {
    if (isWin) {
        return { totalReward: 100, reason: { text: "🏆 Coup gagnant ! (+100 XP)", color: "#2ed573" } };
    }
    if (isDraw) {
        return { totalReward: 2, reason: { text: "🤝 Match nul (+2 XP)", color: "#70a1ff" } };
    }

    let reward = 0;
    let bestReason = null;
    const opponent = (currentPlayer === 1) ? 2 : 1;

    if (lastAction === 3) {
        reward += 2;
        bestReason = { text: "🎯 Contrôle du centre (+2 XP)", color: "#70a1ff" };
    } else if (lastAction === 2 || lastAction === 4) {
        reward += 1;
    }

    if (createdLineOfThree(board, lastAction, currentPlayer)) {
        reward += 10;
        bestReason = { text: "⚡ Menace d'alignement de 3 ! (+10 XP)", color: "#ffa502" };
    }

    if (didBlockOpponentWin(board, lastAction, opponent)) {
        reward += 50;
        bestReason = { text: "🛡️ Parade décisive ! (+50 XP)", color: "#2ed573" };
    }

    return { totalReward: reward, reason: bestReason };
}

function dropToken(g, c, p) {
    if (c < 0 || c >= COLS || g[0][c] !== 0) return null;
    for (let r = ROWS - 1; r >= 0; r--) {
        if (g[r][c] === 0) {
            g[r][c] = p;
            return { r, c };
        }
    }
    return null;
}

// Retourne le tableau des 4 coordonnées gagnantes [[r,c], ...] ou null
function checkWinner(g, p) {
    for (let r = 0; r < 6; r++)
        for (let c = 0; c < 4; c++)
            if (g[r][c] === p && g[r][c+1] === p && g[r][c+2] === p && g[r][c+3] === p)
                return [[r, c], [r, c+1], [r, c+2], [r, c+3]];

    for (let r = 0; r < 3; r++)
        for (let c = 0; c < 7; c++)
            if (g[r][c] === p && g[r+1][c] === p && g[r+2][c] === p && g[r+3][c] === p)
                return [[r, c], [r+1, c], [r+2, c], [r+3, c]];

    for (let r = 3; r < 6; r++)
        for (let c = 0; c < 4; c++)
            if (g[r][c] === p && g[r-1][c+1] === p && g[r-2][c+2] === p && g[r-3][c+3] === p)
                return [[r, c], [r-1, c+1], [r-2, c+2], [r-3, c+3]];

    for (let r = 0; r < 3; r++)
        for (let c = 0; c < 4; c++)
            if (g[r][c] === p && g[r+1][c+1] === p && g[r+2][c+2] === p && g[r+3][c+3] === p)
                return [[r, c], [r+1, c+1], [r+2, c+2], [r+3, c+3]];

    return null;
}

function didBlockOpponentWin(g, col, opponent) {
    let r = 0;
    while (r < ROWS && g[r][col] === 0) r++;
    if (r === ROWS) return false;

    const originalColor = g[r][col];
    g[r][col] = opponent;
    const blocked = checkWinner(g, opponent) !== null;
    g[r][col] = originalColor;

    return blocked;
}

function createdLineOfThree(g, col, player) {
    let r = 0;
    while (r < ROWS && g[r][col] === 0) r++;
    if (r === ROWS) return false;

    let linesOfThree = 0;
    const checkWindow = (r1, c1, r2, c2, r3, c3, r4, c4) => {
        if (r1 < 0 || r1 >= ROWS || c1 < 0 || c1 >= COLS ||
            r4 < 0 || r4 >= ROWS || c4 < 0 || c4 >= COLS) return;

        const window = [g[r1][c1], g[r2][c2], g[r3][c3], g[r4][c4]];
        const playerCount = window.filter(v => v === player).length;
        const emptyCount = window.filter(v => v === 0).length;

        if (playerCount === 3 && emptyCount === 1) linesOfThree++;
    };

    for (let i = 0; i < 4; i++) {
        checkWindow(r, col-3+i, r, col-2+i, r, col-1+i, r, col+i);
        checkWindow(r+i, col, r+1+i, col, r+2+i, col, r+3+i, col);
        checkWindow(r-3+i, col-3+i, r-2+i, col-2+i, r-1+i, col-1+i, r+i, col+i);
        checkWindow(r-3+i, col+3-i, r-2+i, col+2-i, r-1+i, col+1-i, r+i, col-i);
    }

    return linesOfThree > 0;
}

// ==========================================
// 8. BOUTONS ET CONTRÔLES
// ==========================================

document.getElementById('btn-reset').onclick = () => {
    stopArenaRequested = true;
    stopTrainingRequested = true;
    isArena = false;
    isTraining = false;
    lastAITurnContext = null;
    initBoard();
    renderBoard();
    setStatus("Nouvelle partie contre l'IA-A ! Pose un pion rouge.");
};

function toggleArena(btnId, originalText) {
    const btn = document.getElementById(btnId);
    if (isArena && !stopArenaRequested) {
        stopArenaRequested = true;
        btn.innerText = "Arrêt de l'arène...";
    } else if (!isArena && !isTraining) {
        btn.innerText = "🛑 Stopper l'Arène";
        runArena().then(() => { btn.innerText = originalText; });
    }
}

function toggleTraining(aiName, btnId, originalText) {
    const btn = document.getElementById(btnId);
    if (isTraining && !stopTrainingRequested) {
        stopTrainingRequested = true;
        btn.innerText = "Arrêt en cours...";
    } else if (!isTraining && !isArena) {
        btn.innerText = `🛑 Stopper IA-${aiName}`;
        runTraining(aiName).then(() => { btn.innerText = originalText; });
    }
}

document.getElementById('btn-train-a').onclick = () => toggleTraining('A', 'btn-train-a', "💪 Entraîner IA-A (Solo)");
document.getElementById('btn-train-b').onclick = () => toggleTraining('B', 'btn-train-b', "💪 Entraîner IA-B (Solo)");
document.getElementById('btn-arena').onclick = () => toggleArena('btn-arena', "⚔️ ARÈNE : IA-A vs IA-B ⚔️");

document.getElementById('btn-speed').onclick = (e) => {
    isSpectatorMode = !isSpectatorMode;
    const btn = e.currentTarget;
    if (isSpectatorMode) {
        btn.classList.add('spectator');
        btn.innerText = "🍿 Vitesse : Mode Spectateur";
        showFeedbackPopup("🍿 Mode Spectateur : animations en direct activées", "#eccc68");
    } else {
        btn.classList.remove('spectator');
        btn.innerText = "⚡ Vitesse : Mode Turbo";
        showFeedbackPopup("⚡ Mode Turbo : calcul accéléré", "#70a1ff");
    }
};

document.getElementById('btn-wipe').onclick = () => resetBrainsToZero();

// Démarrage initial
initBoard();
initDOMGrid();
initIA();