const ROWS = 6, COLS = 7;
let board = [];
let model, targetModel; // Deux réseaux pour la stabilité
let isTraining = false;

// Paramètres DQN
const GAMMA = 0.95; // Importance du futur
const MEMORY_SIZE = 5000;
let memory = []; // Le Replay Buffer
let totalGames = parseInt(localStorage.getItem('dqn_totalGames')) || 0;
let aiWins = parseInt(localStorage.getItem('dqn_aiWins')) || 0;

// 1. ARCHITECTURE CNN POUR DQN
async function initIA() {
    const createModel = () => {
        const m = tf.sequential();
        m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        m.add(tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({units: 128, activation: 'relu'}));
        m.add(tf.layers.dense({units: 7, activation: 'linear'})); // Sortie = Q-Values
        m.compile({optimizer: tf.train.adam(0.00025), loss: 'meanSquaredError'});
        return m;
    };

    try {
        model = await tf.loadLayersModel('localstorage://dqn-v1');
        targetModel = await tf.loadLayersModel('localstorage://dqn-v1');
        document.getElementById('ia-status').innerText = "DQN Chargé";
    } catch (e) {
        model = createModel();
        targetModel = createModel();
        document.getElementById('ia-status').innerText = "Nouveau DQN";
    }
    updateStatsDisplay();
    renderBoard();
}

// 2. GESTION DE LA MÉMOIRE (Experience Replay)
function remember(state, action, reward, nextState, done) {
    memory.push({state, action, reward, nextState, done});
    if (memory.length > MEMORY_SIZE) memory.shift();
}

// 3. LOGIQUE DE JEU & ANTICIPATION
function getBestMove(grid, epsilon = 0) {
    // Toujours bloquer ou gagner mathématiquement avant de demander au réseau
    for (let c = 0; c < COLS; c++) {
        let t = grid.map(r => [...r]);
        if (dropToken(t, c, 2) && checkWinner(t, 2)) return c;
        t = grid.map(r => [...r]);
        if (dropToken(t, c, 1) && checkWinner(t, 1)) return c;
    }

    if (Math.random() < epsilon) return Math.floor(Math.random() * COLS);

    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

// 4. ENTRAÎNEMENT PAR REPLAY (Le coeur du DQN)
async function trainBatch(size = 64) {
    if (memory.length < size) return;

    const batch = [];
    for(let i=0; i<size; i++) batch.push(memory[Math.floor(Math.random() * memory.length)]);

    const states = tf.tensor2d(batch.map(m => m.state));
    const nextStates = tf.tensor2d(batch.map(m => m.nextState));
    
    // On prédit les Q-values actuelles et futures
    const currentQ = model.predict(states);
    const nextQ = targetModel.predict(nextStates);

    const qValues = currentQ.arraySync();
    const nextQValues = nextQ.arraySync();

    batch.forEach((m, i) => {
        let target = m.reward;
        if (!m.done) {
            // Équation de Bellman
            target = m.reward + GAMMA * Math.max(...nextQValues[i]);
        }
        qValues[i][m.action] = target;
    });

    await model.fit(states, tf.tensor2d(qValues), {epochs: 1, silent: true});
    
    states.dispose(); nextStates.dispose(); currentQ.dispose(); nextQ.dispose();
}

// 5. SIMULATION RAFALE DQN
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 100;
    let epsilon = 0.3; // Exploration

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let turn = (i % 2 === 0) ? 1 : 2;
        let winner = 0;

        while (true) {
            let state = board.flat();
            let col = getBestMove(board, epsilon);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            let prevBoard = [...state];
            if (dropToken(board, col, turn)) {
                let done = checkWinner(board, turn) || board.flat().every(v => v !== 0);
                let reward = 0;
                if (done) {
                    winner = turn;
                    reward = (turn === 2) ? 10 : -10; // Récompense forte
                }

                remember(prevBoard, col, reward, board.flat(), done);
                if (done) break;
                turn = (turn === 1) ? 2 : 1;
            } else break;

            if (i % 5 === 0) { renderBoard(); await new Promise(requestAnimationFrame); }
        }

        totalGames++;
        if (winner === 2) aiWins++;
        updateStatsDisplay();

        await trainBatch(64);

        // Toutes les 10 parties, on synchronise le Target Model
        if (i % 10 === 0) {
            targetModel.setWeights(model.getWeights());
            document.getElementById('status').innerText = `DQN Sync : Match ${i}/100`;
        }
    }

    await model.save('localstorage://dqn-v1');
    isTraining = false;
    document.getElementById('status').innerText = "DQN Optimisé !";
    initBoard(); renderBoard();
}

// --- Fonctions utilitaires standards ---
function initBoard() { board = Array(ROWS).fill().map(() => Array(COLS).fill(0)); }
function dropToken(g, c, p) {
    if (c < 0 || c >= COLS || g[0][c] !== 0) return false;
    for (let r = ROWS - 1; r >= 0; r--) { if (g[r][c] === 0) { g[r][c] = p; return true; } }
    return false;
}
function checkWinner(g, p) {
    for (let r=0; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r][c+1]===p && g[r][c+2]===p && g[r][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<7; c++) if (g[r][c]===p && g[r+1][c]===p && g[r+2][c]===p && g[r+3][c]===p) return true;
    for (let r=3; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r-1][c+1]===p && g[r-2][c+2]===p && g[r-3][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r+1][c+1]===p && g[r+2][c+2]===p && g[r+3][c+3]===p) return true;
    return false;
}
function updateStatsDisplay() {
    document.getElementById('total-games').innerText = totalGames;
    const rate = totalGames > 0 ? ((aiWins / totalGames) * 100).toFixed(1) : 0;
    document.getElementById('ai-winrate').innerText = rate;
    localStorage.setItem('dqn_totalGames', totalGames);
    localStorage.setItem('dqn_aiWins', aiWins);
}
async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { totalGames++; updateStatsDisplay(); return alert("Bravo !"); }
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];
            dropToken(board, aiCol, 2);
            renderBoard();
            if (checkWinner(board, 2)) { totalGames++; aiWins++; updateStatsDisplay(); alert("L'IA gagne !"); }
        }, 200);
    }
}

document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = () => model.save('localstorage://dqn-v1');

initBoard(); initIA();
