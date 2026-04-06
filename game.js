const ROWS = 6;
const COLS = 7;
let board = Array(6).fill().map(() => Array(7).fill(0));
let model;
let isTraining = false;

// Initialisation
async function init() {
    try {
        await loadAI(); // Tente de charger
    } catch {
        model = tf.sequential();
        model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [42]}));
        model.add(tf.layers.dense({units: 64, activation: 'relu'}));
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    }
    document.getElementById('status').innerText = "Prêt à jouer !";
    renderBoard();
}

function renderBoard() {
    const container = document.getElementById('board');
    container.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell';
            if (board[r][c] === 1) div.classList.add('player');
            if (board[r][c] === 2) div.classList.add('ai');
            div.onclick = () => playerMove(c);
            container.appendChild(div);
        }
    }
}

function resetGame() {
    board = Array(6).fill().map(() => Array(7).fill(0));
    renderBoard();
    document.getElementById('status').innerText = "Grille réinitialisée.";
}

// IA : On ajoute tf.nextFrame() pour ne pas freezer l'UI
async function startSelfPlay() {
    if(isTraining) return;
    isTraining = true;
    document.getElementById('status').innerText = "Entraînement en cours (0%)...";
    
    for (let i = 0; i < 50; i++) {
        await selfPlaySingleGame();
        if (i % 5 === 0) {
            document.getElementById('status').innerText = `Entraînement : ${i*2}%`;
            await tf.nextFrame(); // Libère le navigateur pour mettre à jour l'UI
        }
    }
    isTraining = false;
    resetGame();
    document.getElementById('status').innerText = "Entraînement terminé !";
}

async function selfPlaySingleGame() {
    let tempBoard = Array(6).fill().map(() => Array(7).fill(0));
    let history = [];
    let turn = 1;
    let winner = 0;

    for (let moves = 0; moves < 42; moves++) {
        let col = Math.random() < 0.2 ? Math.floor(Math.random() * 7) : getBestMove(tempBoard);
        if (dropToken(tempBoard, col, turn)) {
            history.push({state: tempBoard.flat(), move: col, player: turn});
            if (checkWin(tempBoard, turn)) { winner = turn; break; }
            turn = turn === 1 ? 2 : 1;
        }
    }
    if (winner > 0) await trainFromHistory(history, winner);
}

function getBestMove(state) {
    return tf.tidy(() => {
        const input = tf.tensor2d([state.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

function dropToken(grid, col, p) {
    for (let r = ROWS - 1; r >= 0; r--) {
        if (grid[r][col] === 0) {
            grid[r][col] = p;
            return true;
        }
    }
    return false;
}

function checkWin(grid, p) {
    // Logique simplifiée (Horizontale & Verticale pour l'exemple)
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS - 3; c++) {
            if (grid[r][c]===p && grid[r][c+1]===p && grid[r][c+2]===p && grid[r][c+3]===p) return true;
        }
    }
    for (let r = 0; r < ROWS - 3; r++) {
        for (let c = 0; c < COLS; c++) {
            if (grid[r][c]===p && grid[r+1][c]===p && grid[r+2][c]===p && grid[r+3][c]===p) return true;
        }
    }
    return false;
}

async function trainFromHistory(history, winner) {
    const states = history.map(h => h.state);
    const targets = history.map(h => {
        let t = new Array(7).fill(0);
        t[h.move] = (h.player === winner) ? 1 : -1;
        return t;
    });
    await model.fit(tf.tensor2d(states), tf.tensor2d(targets), {epochs: 1});
}

async function saveAI() {
    await model.save('localstorage://c4-model');
    alert("Mémoire sauvegardée !");
}

async function loadAI() {
    model = await tf.loadLayersModel('localstorage://c4-model');
    model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
}

async function playerMove(col) {
    if (isTraining) return;
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWin(board, 1)) { alert("Gagné !"); return; }
        
        setTimeout(() => {
            let aiCol = getBestMove(board);
            dropToken(board, aiCol, 2);
            renderBoard();
            if (checkWin(board, 2)) alert("L'IA a gagné !");
        }, 300);
    }
}

init();
