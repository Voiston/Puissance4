const ROWS = 6, COLS = 7;
let board = [];
let model, targetModel;
let isTraining = false;

const GAMMA = 0.96;
const MEMORY_SIZE = 3000;
let memory = []; 

let totalGames = parseInt(localStorage.getItem('dqn_pure_totalGames')) || 0;
let aiWins = parseInt(localStorage.getItem('dqn_pure_aiWins')) || 0;

function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const fragment = document.createDocumentFragment();
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell' + (board[r][c] === 1 ? ' player' : board[r][c] === 2 ? ' ai' : '');
            div.onclick = () => handleMove(c);
            fragment.appendChild(div);
        }
    }
    gridEl.appendChild(fragment);
}

// 1. CERVEAU : AUCUNE LOGIQUE DE VICTOIRE CODÉE
function getBestMove(grid, epsilon = 0) {
    if (Math.random() < epsilon) {
        const validCols = [0,1,2,3,4,5,6].filter(c => grid[0][c] === 0);
        return validCols[Math.floor(Math.random() * validCols.length)];
    }

    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        const pred = model.predict(input);
        return pred.argMax(1).dataSync()[0];
    });
}

// 2. INITIALISATION
async function initIA() {
    const createModel = () => {
        const m = tf.sequential();
        m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        m.add(tf.layers.conv2d({filters: 48, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 24, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({units: 96, activation: 'relu'}));
        m.add(tf.layers.dense({units: 7, activation: 'linear'}));
        m.compile({optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError'});
        return m;
    };

    try {
        model = await tf.loadLayersModel('localstorage://dqn-pure-mobile');
        targetModel = await tf.loadLayersModel('localstorage://dqn-pure-mobile');
    } catch (e) {
        model = createModel(); targetModel = createModel();
    }
    updateStatsDisplay();
    renderBoard();
}

// 3. LOGIQUE MANUELLE (IA 100% NEURONALE)
async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;

    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) {
            finishGame(1);
            return;
        }

        document.getElementById('status').innerText = "L'IA analyse...";
        await new Promise(r => setTimeout(r, 200));

        // L'IA décide UNIQUEMENT via son réseau
        let aiCol = getBestMove(board, 0); 
        
        // Sécurité technique (pas stratégique) : si le réseau choisit une colonne pleine
        if (board[0][aiCol] !== 0) {
            aiCol = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
        }

        if (aiCol !== undefined && dropToken(board, aiCol, 2)) {
            renderBoard();
            if (checkWinner(board, 2)) {
                finishGame(2);
            } else if (board.flat().every(v => v !== 0)) {
                finishGame(0);
            } else {
                document.getElementById('status').innerText = "À toi.";
            }
        }
    }
}

// 4. ENTRAÎNEMENT PUR (Mise à jour des poids)
async function trainBatch(size = 32) {
    if (memory.length < size) return;
    tf.tidy(() => {
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
        model.fit(states, tf.tensor2d(qValues), {epochs: 1, silent: true});
    });
}

async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 150; 
    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2;
        while (true) {
            let state = [...board.flat()];
            let epsilon = Math.max(0.1, 0.4 - (i/batchSize));
            let col = getBestMove(board, epsilon);
            
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
            if (col === undefined) break;

            if (dropToken(board, col, turn)) {
                let win = checkWinner(board, turn);
                let done = win || board.flat().every(v => v !== 0);
                
                // Récompenses brutes pour sculpter les neurones
                let reward = win ? (turn === 2 ? 20 : -20) : 0.05;
                
                memory.push({state, action: col, reward, nextState: [...board.flat()], done});
                if (memory.length > MEMORY_SIZE) memory.shift();
                if (done) { if(win && turn === 2) aiWins++; break; }
                turn = (turn === 1) ? 2 : 1;
            } else break;
        }
        totalGames++;
        await trainBatch(32);
        if (i % 15 === 0) {
            targetModel.setWeights(model.getWeights());
            updateStatsDisplay();
            document.getElementById('status').innerText = `Self-Play : ${i}/150`;
            await new Promise(r => setTimeout(r, 50));
        }
    }
    await model.save('localstorage://dqn-pure-mobile');
    isTraining = false;
    document.getElementById('status').innerText = "Cerveau mis à jour.";
    initBoard(); renderBoard();
}

// FONCTIONS TECHNIQUES
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

function finishGame(winner) {
    totalGames++;
    if (winner === 2) aiWins++;
    updateStatsDisplay();
    document.getElementById('status').innerText = winner === 1 ? "Gagné !" : winner === 2 ? "L'IA gagne !" : "Nul !";
}

function updateStatsDisplay() {
    document.getElementById('total-games').innerText = totalGames;
    document.getElementById('ai-winrate').innerText = totalGames > 0 ? ((aiWins/totalGames)*100).toFixed(1) : 0;
    localStorage.setItem('dqn_pure_totalGames', totalGames);
    localStorage.setItem('dqn_pure_aiWins', aiWins);
}

document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); document.getElementById('status').innerText = "À toi."; };
document.getElementById('btn-train').onclick = runTraining;

initBoard(); initIA();
