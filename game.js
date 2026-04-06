const ROWS = 6;
const COLS = 7;
let board = [];
let model;
let isTraining = false;

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

// 1. ARCHITECTURE "DEEP LEARNING" AVANCÉE
async function initIA() {
    try {
        model = await tf.loadLayersModel('localstorage://c4-ultra-model');
        document.getElementById('status').innerText = "IA ULTRA Chargée";
    } catch (e) {
        model = tf.sequential();
        model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [42]}));
        model.add(tf.layers.dropout({rate: 0.1})); // Empêche l'IA de devenir "têtue"
        model.add(tf.layers.dense({units: 128, activation: 'relu'}));
        model.add(tf.layers.dense({units: 64, activation: 'relu'}));
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        model.compile({optimizer: tf.train.adam(0.001), loss: 'meanSquaredError'});
        document.getElementById('status').innerText = "Cerveau ULTRA initialisé.";
    }
    renderBoard();
}

function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    // On utilise un fragment pour optimiser la vitesse d'affichage JS
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

function dropToken(grid, col, player) {
    if (col < 0 || col >= COLS || grid[0][col] !== 0) return false;
    for (let r = ROWS - 1; r >= 0; r--) {
        if (grid[r][col] === 0) {
            grid[r][col] = player;
            return true;
        }
    }
    return false;
}

async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { document.getElementById('status').innerText = "Victoire Humaine !"; return; }
        
        document.getElementById('status').innerText = "L'IA réfléchit...";
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) {
                const free = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0);
                aiCol = free[Math.floor(Math.random() * free.length)];
            }
            if (dropToken(board, aiCol, 2)) {
                renderBoard();
                if (checkWinner(board, 2)) document.getElementById('status').innerText = "L'IA gagne !";
                else document.getElementById('status').innerText = "À vous.";
            }
        }, 200); // Réaction plus rapide pour l'humain aussi
    }
}

function getBestMove(grid) {
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

function checkWinner(grid, p) {
    // Horizontal & Vertical
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS - 3; c++) if (grid[r][c]===p && grid[r][c+1]===p && grid[r][c+2]===p && grid[r][c+3]===p) return true;
    for (let r = 0; r < ROWS - 3; r++) for (let c = 0; c < COLS; c++) if (grid[r][c]===p && grid[r+1][c]===p && grid[r+2][c]===p && grid[r+3][c]===p) return true;
    // Diagonales
    for (let r = 3; r < ROWS; r++) for (let c = 0; c < COLS - 3; c++) if (grid[r][c]===p && grid[r-1][c+1]===p && grid[r-2][c+2]===p && grid[r-3][c+3]===p) return true;
    for (let r = 0; r < ROWS - 3; r++) for (let c = 0; c < COLS - 3; c++) if (grid[r][c]===p && grid[r+1][c+1]===p && grid[r+2][c+2]===p && grid[r+3][c+3]===p) return true;
    return false;
}

// 2. ENTRAÎNEMENT MASSIF VISIBLE (Batch 100 - Vitesse x3)
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 100;
    let totalStates = [];
    let totalLabels = [];

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let moves = [];
        let turn = Math.random() < 0.5 ? 1 : 2;
        let winner = 0;

        document.getElementById('status').innerText = `Hyper-Training : Match ${i}/${batchSize}`;

        for (let step = 0; step < 42; step++) {
            // IA joue (15% hasard pour garder de la flexibilité)
            let col = Math.random() < 0.15 ? Math.floor(Math.random() * 7) : getBestMove(board);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            if (dropToken(board, col, turn)) {
                moves.push({state: board.flat(), move: col, player: turn});
                
                renderBoard(); 
                await sleep(30); // VITESSE X3 (Délai très court)

                if (checkWinner(board, turn)) { winner = turn; break; }
                turn = turn === 1 ? 2 : 1;
            }
        }

        // Collecte des données
        moves.forEach(m => {
            let label = new Array(7).fill(0);
            if (winner === 2 && m.player === 2) label[m.move] = 1; 
            if (winner === 1 && m.player === 2) label[m.move] = -1;
            if (winner === 0) label[m.move] = 0.1;
            totalStates.push(m.state);
            totalLabels.push(label);
        });

        // Entraînement par blocs de 10 matchs pour la fluidité
        if (i % 10 === 0) {
            const xs = tf.tensor2d(totalStates);
            const ys = tf.tensor2d(totalLabels);
            await model.fit(xs, ys, {epochs: 2, shuffle: true});
            xs.dispose(); ys.dispose();
            totalStates = []; totalLabels = [];
        }
        await tf.nextFrame();
    }

    isTraining = false;
    document.getElementById('status').innerText = "100 Matchs terminés ! IA optimisée.";
    initBoard();
    renderBoard();
}

document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); document.getElementById('status').innerText = "Nouvelle partie."; };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = async () => { await model.save('localstorage://c4-ultra-model'); alert("Mémoire ULTRA sauvegardée !"); };

initBoard();
initIA();
