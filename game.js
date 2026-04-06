const ROWS = 6;
const COLS = 7;
const PLAYER = 1;
const AI = 2;

let board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
let model;
let epsilon = 0.3; // Taux d'exploration (30% de coups aléatoires)

// --- 1. INITIALISATION DU MODÈLE ---
async function init() {
    model = tf.sequential();
    model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [ROWS * COLS]}));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: COLS, activation: 'linear'})); // Un score par colonne
    model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    
    console.log("Modèle prêt !");
    renderBoard();
}

// --- 2. LOGIQUE DU JEU ---
function checkWin(p) {
    // Vérification horizontale, verticale et diagonale
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (checkDir(r, c, 0, 1, p) || checkDir(r, c, 1, 0, p) || 
                checkDir(r, c, 1, 1, p) || checkDir(r, c, 1, -1, p)) return true;
        }
    }
    return false;
}

function checkDir(r, c, dr, dc, p) {
    let count = 0;
    for (let i = 0; i < 4; i++) {
        let nr = r + i * dr, nc = c + i * dc;
        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && board[nr][nc] === p) count++;
        else break;
    }
    return count === 4;
}

function dropToken(col, p) {
    for (let r = ROWS - 1; r >= 0; r--) {
        if (board[r][col] === 0) {
            board[r][col] = p;
            return true;
        }
    }
    return false;
}

// --- 3. L'IA ET L'ENTRAÎNEMENT ---
function getBestMove(state) {
    return tf.tidy(() => {
        const input = tf.tensor2d([state.flat()]);
        const prediction = model.predict(input);
        return prediction.argMax(1).dataSync()[0];
    });
}

async function selfPlay(games = 100) {
    document.getElementById('status').innerText = "Entraînement en cours...";
    
    for (let g = 0; g < games; g++) {
        board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
        let turn = PLAYER;
        let history = [];

        while (true) {
            let col = (Math.random() < epsilon) ? Math.floor(Math.random() * COLS) : getBestMove(board);
            
            if (dropToken(col, turn)) {
                history.push({state: [...board.map(r => [...r])], move: col, player: turn});
                
                if (checkWin(turn)) {
                    await trainFromHistory(history, turn); // On récompense le vainqueur
                    break;
                }
                if (board.flat().every(c => c !== 0)) break; // Match nul
                turn = (turn === PLAYER) ? AI : PLAYER;
            }
        }
    }
    document.getElementById('status').innerText = "Entraînement terminé ! À vous de jouer.";
    renderBoard();
}

async function trainFromHistory(history, winner) {
    const states = [];
    const targets = [];

    for (let i = 0; i < history.length; i++) {
        const {state, move, player} = history[i];
        let reward = (player === winner) ? 1.0 : -1.0;
        
        // On récupère la prédiction actuelle
        const input = tf.tensor2d([state.flat()]);
        const target = model.predict(input).dataSync();
        
        // On ajuste la valeur du coup joué
        target[move] = reward; 
        
        states.push(state.flat());
        targets.push(target);
    }

    await model.fit(tf.tensor2d(states), tf.tensor2d(targets), {epochs: 1});
}

// --- 4. INTERFACE ---
function renderBoard() {
    const container = document.getElementById('board');
    container.innerHTML = '';
    board.forEach((row, r) => {
        row.forEach((cell, c) => {
            const div = document.createElement('div');
            div.className = 'cell' + (cell === PLAYER ? ' player' : cell === AI ? ' ai' : '');
            div.onclick = () => playerMove(c);
            container.appendChild(div);
        });
    });
}
// Sauvegarder le cerveau de l'IA
async function saveAI() {
    await model.save('localstorage://connect4-model');
    document.getElementById('status').innerText = "Cerveau de l'IA sauvegardé !";
}

// Charger le cerveau au démarrage
async function loadAI() {
    try {
        model = await tf.loadLayersModel('localstorage://connect4-model');
        // Il faut recompiler après le chargement
        model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
        document.getElementById('status').innerText = "Cerveau chargé avec succès.";
        renderBoard();
    } catch (e) {
        console.log("Aucune sauvegarde trouvée, création d'un nouveau cerveau.");
        init(); // Crée un nouveau modèle si aucun n'existe
    }
}

async function playerMove(col) {
    if (dropToken(col, PLAYER)) {
        renderBoard();
        if (checkWin(PLAYER)) return alert("Bravo !");
        
        // Tour de l'IA
        setTimeout(async () => {
            let aiCol = getBestMove(board);
            if (dropToken(aiCol, AI)) {
                renderBoard();
                if (checkWin(AI)) alert("L'IA a gagné !");
            }
        }, 500);
    }
}

init();

