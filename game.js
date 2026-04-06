const ROWS = 6, COLS = 7;
let board = [];
let model;
let isTraining = false;

// 1. ARCHITECTURE CONVOLUTIONNELLE (Le standard des IA de jeux de plateau)
async function initIA() {
    try {
        model = await tf.loadLayersModel('localstorage://c4-cnn-model');
        document.getElementById('status').innerText = "IA CNN Chargée - Prête au combat";
    } catch (e) {
        model = tf.sequential();
        // La couche Conv2D permet à l'IA de "voir" les lignes de 4
        model.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        model.add(tf.layers.conv2d({filters: 64, kernelSize: 4, activation: 'relu'}));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 128, activation: 'relu'}));
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        model.compile({optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError'});
        document.getElementById('status').innerText = "Cerveau CNN (Vision Spatiale) initialisé.";
    }
    renderBoard();
}

// 2. SIMULATION TURBO (Pas de délai visuel pendant le calcul)
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 100;
    let allStates = [], allLabels = [];

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let moves = [];
        let turn = (i % 2 === 0) ? 1 : 2;
        let winner = 0;

        // Simulation à la vitesse du processeur
        for (let step = 0; step < 42; step++) {
            let col = Math.random() < 0.2 ? Math.floor(Math.random() * 7) : getBestMove(board);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            if (dropToken(board, col, turn)) {
                moves.push({state: board.flat(), move: col, player: turn});
                if (checkWinner(board, turn)) { winner = turn; break; }
                turn = (turn === 1) ? 2 : 1;
            }
        }

        // Attribution des scores (Renforcement positif/négatif)
        moves.forEach(m => {
            let label = new Array(7).fill(0);
            if (winner === 2) label[m.move] = (m.player === 2) ? 1.0 : -1.0;
            if (winner === 1) label[m.move] = (m.player === 2) ? -1.0 : 0.5;
            allStates.push(m.state);
            allLabels.push(label);
        });

        // Mise à jour visuelle tous les 10 matchs pour montrer que ça travaille
        if (i % 10 === 0) {
            renderBoard();
            document.getElementById('status').innerText = `Turbo-Training : ${i}%`;
            await tf.nextFrame(); 
        }
    }

    // Entraînement massif final
    const xs = tf.tensor2d(allStates);
    const ys = tf.tensor2d(allLabels);
    await model.fit(xs, ys, {epochs: 5, shuffle: true, batchSize: 32});
    xs.dispose(); ys.dispose();

    isTraining = false;
    document.getElementById('status').innerText = "100 parties simulées en un clin d'œil !";
    initBoard();
    renderBoard();
}

// --- Les fonctions de base restent identiques mais optimisées ---

function getBestMove(grid) {
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const frag = document.createDocumentFragment();
    board.forEach((row, r) => {
        row.forEach((cell, c) => {
            const div = document.createElement('div');
            div.className = 'cell' + (cell === 1 ? ' player' : cell === 2 ? ' ai' : '');
            div.onclick = () => handleMove(c);
            frag.appendChild(div);
        });
    });
    gridEl.appendChild(frag);
}

function initBoard() { board = Array(6).fill().map(() => Array(7).fill(0)); }

function dropToken(grid, col, p) {
    for (let r = 5; r >= 0; r--) { if (grid[r][col] === 0) { grid[r][col] = p; return true; } }
    return false;
}

function checkWinner(g, p) {
    for (let r=0; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r][c+1]===p && g[r][c+2]===p && g[r][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<7; c++) if (g[r][c]===p && g[r+1][c]===p && g[r+2][c]===p && g[r+3][c]===p) return true;
    for (let r=3; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r-1][c+1]===p && g[r-2][c+2]===p && g[r-3][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r+1][c+1]===p && g[r+2][c+2]===p && g[r+3][c+3]===p) return true;
    return false;
}

async function handleMove(c) {
    if (isTraining || board[0][c] !== 0) return;
    if (dropToken(board, c, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) return alert("Gagné !");
        let aiC = getBestMove(board);
        if (board[0][aiC] !== 0) aiC = [0,1,2,3,4,5,6].filter(x => board[0][x]===0)[0];
        dropToken(board, aiC, 2);
        setTimeout(renderBoard, 50);
        if (checkWinner(board, 2)) setTimeout(() => alert("L'IA gagne !"), 100);
    }
}

document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = () => model.save('localstorage://c4-cnn-model');

initBoard();
initIA();
