const ROWS = 6, COLS = 7;
let board = [];
let model, targetModel;
let isTraining = false;

// PARAMÈTRES RÉDUITS POUR MOBILE
const GAMMA = 0.95;
const MEMORY_SIZE = 2000; // Moins de RAM utilisée
let memory = []; 

let totalGames = parseInt(localStorage.getItem('dqn_pure_totalGames')) || 0;
let aiWins = parseInt(localStorage.getItem('dqn_pure_aiWins')) || 0;

function initBoard() { board = Array(ROWS).fill().map(() => Array(COLS).fill(0)); }

function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell' + (board[r][c] === 1 ? ' player' : board[r][c] === 2 ? ' ai' : '');
            div.onclick = () => handleMove(c);
            gridEl.appendChild(div);
        }
    }
}

async function initIA() {
    const createModel = () => {
        const m = tf.sequential();
        m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        // On réduit à 32 et 16 filtres pour soulager le GPU du portable
        m.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({units: 64, activation: 'relu'}));
        m.add(tf.layers.dense({units: 7, activation: 'linear'}));
        m.compile({optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError'});
        return m;
    };

    try {
        model = await tf.loadLayersModel('localstorage://dqn-mobile-v1');
        targetModel = await tf.loadLayersModel('localstorage://dqn-mobile-v1');
    } catch (e) {
        model = createModel();
        targetModel = createModel();
    }
    updateStatsDisplay();
    renderBoard();
}

function getBestMove(grid, epsilon = 0) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * COLS);
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

async function trainBatch(size = 32) { // Batch plus petit (32 au lieu de 64)
    if (memory.length < size) return;
    
    tf.engine().startScope(); // Nettoyage radical
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
    tf.engine().endScope(); 
}

async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 250; 

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2;
        let epsilon = Math.max(0.1, 0.4 - (i / batchSize));

        while (true) {
            let state = [...board.flat()];
            let col = getBestMove(board, epsilon);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
            if (col === undefined) break;

            if (dropToken(board, col, turn)) {
                let isWin = checkWinner(board, turn);
                let done = isWin || board.flat().every(v => v !== 0);
                let reward = isWin ? (turn === 2 ? 10 : -10) : 0.01;

                memory.push({state, action: col, reward, nextState: [...board.flat()], done});
                if (memory.length > MEMORY_SIZE) memory.shift();

                if (i % 25 === 0) { renderBoard(); await new Promise(requestAnimationFrame); }

                if (done) { if(isWin && turn === 2) aiWins++; break; }
                turn = (turn === 1) ? 2 : 1;
            } else break;
        }

        totalGames++;
        await trainBatch(32);

        if (i % 10 === 0) {
            targetModel.setWeights(model.getWeights());
            updateStatsDisplay();
            document.getElementById('status').innerText = `Mobile Train: ${i}/250`;
            // Pause plus longue pour laisser le téléphone refroidir
            await new Promise(resolve => setTimeout(resolve, 150)); 
        }
        
        // Sauvegarde de sécurité toutes les 50 parties
        if (i % 50 === 0) await model.save('localstorage://dqn-mobile-v1');
    }

    isTraining = false;
    document.getElementById('status').innerText = "Terminé !";
    initBoard(); renderBoard();
}

// Fonctions techniques inchangées
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
    document.getElementById('ai-winrate').innerText = totalGames > 0 ? ((aiWins / totalGames) * 100).toFixed(1) : 0;
    localStorage.setItem('dqn_pure_totalGames', totalGames);
    localStorage.setItem('dqn_pure_aiWins', aiWins);
}
async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { totalGames++; updateStatsDisplay(); return; }
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
            dropToken(board, aiCol, 2);
            renderBoard();
            if (checkWinner(board, 2)) { totalGames++; aiWins++; updateStatsDisplay(); }
        }, 150);
    }
}
document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); };
document.getElementById('btn-train').onclick = runTraining;
initBoard(); initIA();
