const ROWS = 6;
const COLS = 7;
let board = [];
let model;
let isTraining = false;

// Chargement des statistiques depuis le stockage local
let totalGames = parseInt(localStorage.getItem('p4_totalGames')) || 0;
let aiWins = parseInt(localStorage.getItem('p4_aiWins')) || 0;

// 1. INITIALISATION DU CERVEAU (CNN)
async function initIA() {
    try {
        model = await tf.loadLayersModel('localstorage://c4-ultra-hardcore');
        document.getElementById('ia-status').innerText = "Expert CNN";
    } catch (e) {
        model = tf.sequential();
        model.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        model.add(tf.layers.conv2d({filters: 64, kernelSize: 4, activation: 'relu', padding: 'same'}));
        model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 128, activation: 'relu'}));
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        model.compile({optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError'});
        document.getElementById('ia-status').innerText = "Nouveau Cerveau";
    }
    updateStatsDisplay();
    renderBoard();
}

// 2. STATISTIQUES ET AFFICHAGE
function updateStatsDisplay() {
    document.getElementById('total-games').innerText = totalGames;
    const rate = totalGames > 0 ? ((aiWins / totalGames) * 100).toFixed(1) : 0;
    document.getElementById('ai-winrate').innerText = rate;
    
    localStorage.setItem('p4_totalGames', totalGames);
    localStorage.setItem('p4_aiWins', aiWins);
}

function renderBoard() {
    const gridEl = document.getElementById('board');
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

// 3. INTELLIGENCE ET ANTICIPATION
function getBestMove(grid) {
    // Règle d'or 1 : Gagner si possible
    for (let c = 0; c < COLS; c++) {
        let temp = grid.map(r => [...r]);
        if (dropToken(temp, c, 2) && checkWinner(temp, 2)) return c;
    }
    // Règle d'or 2 : Bloquer l'humain
    for (let c = 0; c < COLS; c++) {
        let temp = grid.map(r => [...r]);
        if (dropToken(temp, c, 1) && checkWinner(temp, 1)) return c;
    }
    // Sinon : Réseau de neurones
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

// 4. ENTRAÎNEMENT RAFALE AVEC AUTO-SAVE
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 100;
    let allStates = [], allLabels = [];

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let moves = [], winner = 0;
        let turn = (Math.random() < 0.5) ? 1 : 2;

        document.getElementById('status').innerText = `Simulation Match ${i}/${batchSize}...`;

        for (let step = 0; step < 42; step++) {
            let col = Math.random() < 0.2 ? Math.floor(Math.random() * 7) : getBestMove(board);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            if (dropToken(board, col, turn)) {
                moves.push({state: board.flat(), move: col, player: turn});
                renderBoard();
                await new Promise(requestAnimationFrame);
                if (checkWinner(board, turn)) { winner = turn; break; }
                turn = (turn === 1) ? 2 : 1;
            }
        }

        totalGames++;
        if (winner === 2) aiWins++;
        updateStatsDisplay();

        // Préparation apprentissage
        moves.forEach(m => {
            let label = new Array(7).fill(0);
            if (winner === 2) label[m.move] = (m.player === 2) ? 1.0 : -0.5;
            if (winner === 1) label[m.move] = (m.player === 2) ? -2.0 : 0.5;
            allStates.push(m.state); allLabels.push(label);
        });

        // Entraînement et AUTO-SAVE toutes les 20 parties
        if (i % 10 === 0) {
            const xs = tf.tensor2d(allStates);
            const ys = tf.tensor2d(allLabels);
            await model.fit(xs, ys, {epochs: 1});
            xs.dispose(); ys.dispose();
            allStates = []; allLabels = [];
            
            if (i % 20 === 0) {
                await model.save('localstorage://c4-ultra-hardcore');
                console.log("Auto-save effectué");
            }
        }
    }
    isTraining = false;
    document.getElementById('status').innerText = "Entraînement terminé !";
    initBoard(); renderBoard();
}

// 5. LOGIQUE DE JEU CLASSIQUE
function initBoard() { board = Array(ROWS).fill().map(() => Array(COLS).fill(0)); }

function dropToken(grid, col, p) {
    if (col < 0 || col >= COLS || grid[0][col] !== 0) return false;
    for (let r = ROWS - 1; r >= 0; r--) { if (grid[r][col] === 0) { grid[r][col] = p; return true; } }
    return false;
}

function checkWinner(g, p) {
    for (let r=0; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r][c+1]===p && g[r][c+2]===p && g[r][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<7; c++) if (g[r][c]===p && g[r+1][c]===p && g[r+2][c]===p && g[r+3][c]===p) return true;
    for (let r=3; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r-1][c+1]===p && g[r-2][c+2]===p && g[r-3][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r+1][c+1]===p && g[r+2][c+2]===p && g[r+3][c+3]===p) return true;
    return false;
}

async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { 
            totalGames++; updateStatsDisplay();
            document.getElementById('status').innerText = "Gagné !"; return; 
        }
        document.getElementById('status').innerText = "L'IA réfléchit...";
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];
            if (dropToken(board, aiCol, 2)) {
                renderBoard();
                if (checkWinner(board, 2)) {
                    totalGames++; aiWins++; updateStatsDisplay();
                    document.getElementById('status').innerText = "L'IA gagne !";
                } else { document.getElementById('status').innerText = "À vous."; }
            }
        }, 300);
    }
}

// BOUTONS
document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); document.getElementById('status').innerText = "Partie prête."; };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = async () => { await model.save('localstorage://c4-ultra-hardcore'); alert("Sauvegarde réussie !"); };

initBoard();
initIA();
