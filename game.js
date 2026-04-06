const ROWS = 6, COLS = 7;
let board = [];
let model, targetModel;
let isTraining = false;

// Paramètres DQN
const GAMMA = 0.97;
const MEMORY_SIZE = 10000;
let memory = []; 

let totalGames = parseInt(localStorage.getItem('dqn_pure_totalGames')) || 0;
let aiWins = parseInt(localStorage.getItem('dqn_pure_aiWins')) || 0;

// 1. INITIALISATION DU PLATEAU (Indispensable avant tout)
function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

// 2. RENDU VISUEL
function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const fragment = document.createDocumentFragment();
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell';
            if (board[r][c] === 1) div.classList.add('player');
            if (board[r][c] === 2) div.classList.add('ai');
            div.onclick = () => handleMove(c);
            fragment.appendChild(div);
        }
    }
    gridEl.appendChild(fragment);
}

// 3. ARCHITECTURE DU CERVEAU
async function initIA() {
    const createModel = () => {
        const m = tf.sequential();
        m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        m.add(tf.layers.conv2d({filters: 128, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({units: 256, activation: 'relu'}));
        m.add(tf.layers.dense({units: 7, activation: 'linear'}));
        m.compile({optimizer: tf.train.adam(0.0001), loss: 'meanSquaredError'});
        return m;
    };

    try {
        model = await tf.loadLayersModel('localstorage://dqn-pure-v1');
        targetModel = await tf.loadLayersModel('localstorage://dqn-pure-v1');
        document.getElementById('ia-status').innerText = "Mode Pur : Chargé";
    } catch (e) {
        model = createModel();
        targetModel = createModel();
        document.getElementById('ia-status').innerText = "Mode Pur : Initialisé";
    }
    updateStatsDisplay();
    renderBoard(); // On affiche la grille une fois l'IA prête
}

// 4. LOGIQUE DQN (SANS RÈGLES HARDCODÉES)
function getBestMove(grid, epsilon = 0) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * COLS);
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

function remember(state, action, reward, nextState, done) {
    memory.push({state, action, reward, nextState, done});
    if (memory.length > MEMORY_SIZE) memory.shift();
}

async function trainBatch(size = 128) {
    if (memory.length < size) return;
    const batch = [];
    for(let i=0; i<size; i++) batch.push(memory[Math.floor(Math.random() * memory.length)]);
    const states = tf.tensor2d(batch.map(m => m.state));
    const nextStates = tf.tensor2d(batch.map(m => m.nextState));
    const currentQ = model.predict(states);
    const nextQ = targetModel.predict(nextStates);
    const qValues = currentQ.arraySync();
    const nextQValues = nextQ.arraySync();
    batch.forEach((m, i) => {
        let target = m.reward;
        if (!m.done) target = m.reward + GAMMA * Math.max(...nextQValues[i]);
        qValues[i][m.action] = target;
    });
    await model.fit(states, tf.tensor2d(qValues), {epochs: 1, silent: true});
    tf.dispose([states, nextStates, currentQ, nextQ]);
}

// 5. SIMULATION
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 250; 
    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2;
        let winner = 0;
        let epsilon = Math.max(0.05, 0.4 - (i / batchSize));

        while (true) {
            let state = [...board.flat()];
            let col = getBestMove(board, epsilon);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            if (dropToken(board, col, turn)) {
                let isWin = checkWinner(board, turn);
                let isDraw = !isWin && board.flat().every(v => v !== 0);
                let done = isWin || isDraw;
                let reward = 0.1;
                if (isWin) reward = (turn === 2) ? 15 : -15;
                if (isDraw) reward = 2;

                remember(state, col, reward, board.flat(), done);
                if (done) { winner = isWin ? turn : 0; break; }
                turn = (turn === 1) ? 2 : 1;
            } else break;
        }

        totalGames++;
        if (winner === 2) aiWins++;
        updateStatsDisplay();
        await trainBatch(128);

        if (i % 10 === 0) {
            targetModel.setWeights(model.getWeights());
            document.getElementById('status').innerText = `Apprentissage : ${i}/250`;
            renderBoard();
            await new Promise(requestAnimationFrame);
        }
    }
    await model.save('localstorage://dqn-pure-v1');
    isTraining = false;
    document.getElementById('status').innerText = "Cycle terminé.";
    initBoard(); renderBoard();
}

// 6. FONCTIONS TECHNIQUES
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
    localStorage.setItem('dqn_pure_totalGames', totalGames);
    localStorage.setItem('dqn_pure_aiWins', aiWins);
}

async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { totalGames++; updateStatsDisplay(); document.getElementById('status').innerText = "Gagné !"; return; }
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];
            dropToken(board, aiCol, 2);
            renderBoard();
            if (checkWinner(board, 2)) { totalGames++; aiWins++; updateStatsDisplay(); document.getElementById('status').innerText = "L'IA gagne !"; }
        }, 200);
    }
}

document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = () => model.save('localstorage://dqn-pure-v1');

// Lancement
initBoard();
initIA();
