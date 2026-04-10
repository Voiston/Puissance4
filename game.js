const ROWS = 6, COLS = 7;
let board = [];
let isTraining = false;
let isArena = false;
let stopTrainingRequested = false;
let isProcessingMove = false; // Verrou de sécurité anti-autoclicker
let lastAITurnContext = null; // Permet de punir l'IA avec précision

// Paramètres DQN
const GAMMA = 0.99; 
const MEMORY_SIZE = 60000; // Taille idéale pour la réactivité face à un humain

const AIs = {
    'A': { model: null, target: null, memory: [], storage: 'localstorage://dqn-ia-a' },
    'B': { model: null, target: null, memory: [], storage: 'localstorage://dqn-ia-b' }
};

// 1. INITIALISATION DU PLATEAU ET RENDU VISUEL
function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

function renderBoard() {
    if (isTraining) return; 
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const fragment = document.createDocumentFragment();
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell' + (board[r][c] === 1 ? ' player' : board[r][c] === 2 ? ' ai' : '');
            div.onclick = () => { if(!isArena) handleMove(c); };
            fragment.appendChild(div);
        }
    }
    gridEl.appendChild(fragment);
}

// 2. CRÉATION DES DEUX CERVEAUX
async function initIA() {
    const compileModel = (m) => {
        m.compile({optimizer: tf.train.adam(0.00025), loss: 'meanSquaredError'});
        return m;
    };

   const createModel = () => {
    const m = tf.sequential();
    m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
    
    // On garde les convolutions (elles sont légères et puissantes pour la vision)
    m.add(tf.layers.conv2d({filters: 64, kernelSize: 4, activation: 'relu', padding: 'same'}));
    m.add(tf.layers.conv2d({filters: 64, kernelSize: 4, activation: 'relu', padding: 'same'}));
    
    m.add(tf.layers.flatten());
    
    // C'est ici qu'on réduit drastiquement pour gagner de la place
    m.add(tf.layers.dense({units: 128, activation: 'relu'})); // Passage de 512 à 128
    m.add(tf.layers.dense({units: 64, activation: 'relu'}));  // Passage de 256 à 64
    m.add(tf.layers.dense({units: 7, activation: 'linear'}));
    
    return compileModel(m); 
};

    try { 
        AIs['A'].model = await tf.loadLayersModel(AIs['A'].storage); 
        AIs['A'].target = await tf.loadLayersModel(AIs['A'].storage); 
        compileModel(AIs['A'].model); compileModel(AIs['A'].target);
    } catch (e) { 
        AIs['A'].model = createModel(); AIs['A'].target = createModel(); 
    }

    try { 
        AIs['B'].model = await tf.loadLayersModel(AIs['B'].storage); 
        AIs['B'].target = await tf.loadLayersModel(AIs['B'].storage); 
        compileModel(AIs['B'].model); compileModel(AIs['B'].target);
    } catch (e) { 
        AIs['B'].model = createModel(); AIs['B'].target = createModel(); 
    }

    document.getElementById('ia-status').innerText = "IA A et B prêtes (Mode Double DQN + Symétrie)";
    renderBoard();
}

// 3. PRÉDICTION SÉCURISÉE
function getBestMove(grid, aiName, epsilon = 0) {
    if (Math.random() < epsilon) {
        const validCols = [0,1,2,3,4,5,6].filter(c => grid[0][c] === 0);
        return validCols[Math.floor(Math.random() * validCols.length)];
    }

    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        const pred = AIs[aiName].model.predict(input);
        return pred.argMax(1).dataSync()[0];
    });
}

// 🌟 NOUVEAU : GESTIONNAIRE DE MÉMOIRE (SYMÉTRIE) 🌟
function getMirroredState(flatBoard) {
    let mirrored = new Array(42);
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            mirrored[r * COLS + c] = flatBoard[r * COLS + (COLS - 1 - c)];
        }
    }
    return mirrored;
}

function saveMemory(aiName, state, action, reward, nextState, done, clones = 1) {
    const memory = AIs[aiName].memory;
    const mirroredState = getMirroredState(state);
    const mirroredNextState = getMirroredState(nextState);
    const mirroredAction = (COLS - 1) - action; 

    for (let i = 0; i < clones; i++) {
        memory.push({ state, action, reward, nextState, done });
        // Bonus Symétrie gratuit !
        memory.push({ state: mirroredState, action: mirroredAction, reward, nextState: mirroredNextState, done });
    }

    while (memory.length > MEMORY_SIZE) memory.shift();
}

// 🌟 NOUVEAU : ENTRAÎNEMENT AVEC DOUBLE DQN 🌟
async function trainBatch(aiName, size = 512) {
    const memory = AIs[aiName].memory;
    if (memory.length < size) return;

    const batch = [];
    for(let i=0; i<size; i++) batch.push(memory[Math.floor(Math.random() * memory.length)]);

    const { x, y } = tf.tidy(() => {
        const states = tf.tensor2d(batch.map(m => m.state));
        const nextStates = tf.tensor2d(batch.map(m => m.nextState));
        
        const currentQ = AIs[aiName].model.predict(states);
        
        // Double Cerveau
        const nextQMain = AIs[aiName].model.predict(nextStates);
        const nextQTarget = AIs[aiName].target.predict(nextStates);
        
        const qValues = currentQ.arraySync();
        const nextQMainValues = nextQMain.arraySync();
        const nextQTargetValues = nextQTarget.arraySync();

        batch.forEach((m, i) => {
            let target = m.reward;
            if (!m.done) {
                let bestFutureAction = nextQMainValues[i].indexOf(Math.max(...nextQMainValues[i]));
                let futureValue = nextQTargetValues[i][bestFutureAction];
                target = m.reward + GAMMA * futureValue;
            }
            qValues[i][m.action] = target;
        });

        return { x: states, y: tf.tensor2d(qValues) };
    });

    await AIs[aiName].model.fit(x, y, {epochs: 1, silent: true});
    x.dispose(); y.dispose();
}

// 5. BOUCLE D'ENTRAÎNEMENT INVISIBLE
async function runTraining(aiName) {
    if (isTraining || isArena || isProcessingMove) return;
    isTraining = true;
    stopTrainingRequested = false; 
    
    let gamesPlayed = 0; 
    const statusText = document.getElementById('status');
    statusText.innerText = `Entraînement IA-${aiName} démarré...`;

    while (!stopTrainingRequested) {
        gamesPlayed++;
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2; 
        
        let epsilon = Math.max(0.10, 0.8 - (gamesPlayed / 10000));

        while (true) {
            let state = [...board.flat()];
            let col = getBestMove(board, aiName, epsilon);
            
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
            if (col === undefined) break;

            if (dropToken(board, col, turn)) {
                let win = checkWinner(board, turn);
                let done = win || board.flat().every(v => v !== 0);
                
                let reward = calculateReward(board, col, turn, win, done && !win);

                // Sauvegarde optimisée (réelle + miroir)
                saveMemory(aiName, state, col, reward, [...board.flat()], done, 1);
                
                if (done) break;
                turn = (turn === 1) ? 2 : 1;
            } else break;
        }

        await trainBatch(aiName, 512); 

        if (gamesPlayed % 10 === 0) {
            statusText.innerText = `Entraînement IA-${aiName} : ${gamesPlayed} parties jouées (Clique pour stopper)`;
        }

        if (gamesPlayed % 200 === 0) {
            AIs[aiName].target.setWeights(AIs[aiName].model.getWeights());
            await AIs[aiName].model.save(AIs[aiName].storage); 
        }
        
        await tf.nextFrame(); 
    }

    await AIs[aiName].model.save(AIs[aiName].storage);
    isTraining = false;
    statusText.innerText = `Entraînement stoppé ! IA-${aiName} optimisée sur ${gamesPlayed} parties.`;
    initBoard(); renderBoard();
}

// 6. L'ARÈNE ÉVOLUTIVE (Combat avec apprentissage)
async function runArena() {
    if (isTraining || isArena || isProcessingMove) return;
    isArena = true;
    initBoard(); renderBoard();
    
    document.getElementById('status').innerText = "⚔️ COMBAT ÉVOLUTIF : IA-A vs IA-B ⚔️";
    await new Promise(r => setTimeout(r, 1000));

    let turn = 1; 
    let history = []; // On stocke le film du match pour l'analyser à la fin

    while (true) {
        let activeAI = turn === 1 ? 'A' : 'B';
        let opponentAI = turn === 1 ? 'B' : 'A';
        
        let stateBefore = [...board.flat()];
        let col = getBestMove(board, activeAI, 0.05); // Petit Epsilon (5%) pour varier les matchs
        
        if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
        
        if (col === undefined) {
            document.getElementById('status').innerText = "Match Nul !";
            break;
        }

        if (dropToken(board, col, turn)) {
            renderBoard();
            
            let win = checkWinner(board, turn);
            let draw = board.flat().every(v => v !== 0);
            let done = win || draw;

            // On note le mouvement dans l'historique du match
            history.push({
                aiName: activeAI,
                state: stateBefore,
                action: col,
                nextState: [...board.flat()],
                done: done
            });

            if (done) {
                // --- PHASE D'APPRENTISSAGE POST-MATCH ---
                if (win) {
                    document.getElementById('status').innerText = `🏆 L'IA-${activeAI} gagne et apprend !`;
                    
                    // 1. On récompense le vainqueur (Poids faible : x5 -> x10 avec symétrie)
                    let winnerMove = history[history.length - 1];
                    saveMemory(activeAI, winnerMove.state, winnerMove.action, 100, winnerMove.nextState, true, 5);
                    
                    // 2. On punit le perdant (Poids faible : x5 -> x10 avec symétrie)
                    // Le perdant est celui qui a joué juste avant le dernier coup
                    if (history.length >= 2) {
                        let loserMove = history[history.length - 2];
                        saveMemory(opponentAI, loserMove.state, loserMove.action, -100, loserMove.nextState, true, 5);
                    }
                } else {
                    document.getElementById('status').innerText = "Égalité !";
                }

                // 3. On lance un micro-entraînement pour les deux
                await trainBatch('A', 256);
                await trainBatch('B', 256);
                
                // 4. On synchronise et on sauvegarde
                AIs['A'].target.setWeights(AIs['A'].model.getWeights());
                AIs['B'].target.setWeights(AIs['B'].model.getWeights());
                await AIs['A'].model.save(AIs['A'].storage);
                await AIs['B'].model.save(AIs['B'].storage);
                
                break;
            }

            turn = (turn === 1) ? 2 : 1;
            await new Promise(r => setTimeout(r, 100)); // Vitesse de l'arène
        } else break;
    }
    isArena = false;
}

// 7. JOUER MANUELLEMENT (Avec Lock, Symétrie et Double DQN)
async function handleMove(col) {
    if (isTraining || isArena || isProcessingMove || board[0][col] !== 0) return;

    isProcessingMove = true; // 🔒 VERROU : On bloque les clics intempestifs

    try {
        if (dropToken(board, col, 1)) { 
            renderBoard();
            let humanWon = checkWinner(board, 1);
            let isDraw = board.flat().every(v => v !== 0);

            // SI L'HUMAIN GAGNE : PUNITION CHIRURGICALE
            if (humanWon || isDraw) { 
                document.getElementById('status').innerText = humanWon ? "Tu as battu l'IA-A !" : "Nul !"; 
                
                if (humanWon && lastAITurnContext) {
                    // On punit l'IA x25 (qui devient x50 grâce à la symétrie) sur son dernier coup précis
                    saveMemory('A', lastAITurnContext.state, lastAITurnContext.action, -100, lastAITurnContext.nextState, true, 25);
                    
                    await trainBatch('A', 256);
                    await trainBatch('A', 256);
                    
                    AIs['A'].target.setWeights(AIs['A'].model.getWeights());
                    await AIs['A'].model.save(AIs['A'].storage);
                }
                lastAITurnContext = null;
                return; 
            }

            document.getElementById('status').innerText = "L'IA-A réfléchit...";
            await new Promise(r => setTimeout(r, 50)); 

            // --- TOUR DE L'IA ---
            let stateBeforeAI = [...board.flat()]; 
            
            let aiCol = getBestMove(board, 'A', 0); 
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);

            if (aiCol !== undefined && dropToken(board, aiCol, 2)) { 
                renderBoard();
                
                let aiWon = checkWinner(board, 2);
                let aiDraw = board.flat().every(v => v !== 0);
                let done = aiWon || aiDraw;

                let reward = calculateReward(board, aiCol, 2, aiWon, aiDraw);

                // On stocke le contexte exact pour la punition potentielle au tour d'après
                lastAITurnContext = { state: stateBeforeAI, action: aiCol, nextState: [...board.flat()] };

                // Oversampling de ses parties avec l'humain (clones = 25 -> 50 avec symétrie)
                saveMemory('A', stateBeforeAI, aiCol, reward, [...board.flat()], done, 25);

                await trainBatch('A', 256);
                await trainBatch('A', 256);
                await trainBatch('A', 256);

                if (aiWon) document.getElementById('status').innerText = "L'IA-A t'a écrasé !";
                else if (aiDraw) document.getElementById('status').innerText = "Nul !";
                else document.getElementById('status').innerText = "À toi.";

                if (done) {
                    AIs['A'].target.setWeights(AIs['A'].model.getWeights());
                    await AIs['A'].model.save(AIs['A'].storage);
                    lastAITurnContext = null;
                }
            }
        }
    } finally {
        isProcessingMove = false; // 🔓 DÉVERROUILLAGE : Toujours rouvrir la porte
    }
}

// 8. FONCTION DE RÉCOMPENSE AVANCÉE
function calculateReward(board, lastAction, currentPlayer, isWin, isDraw) {
    if (isWin) return 100;
    if (isDraw) return 2;

    let reward = 0;
    const opponent = (currentPlayer === 1) ? 2 : 1;

    if (lastAction === 3) reward += 2; 
    else if (lastAction === 2 || lastAction === 4) reward += 1; 

    if (didBlockOpponentWin(board, lastAction, opponent)) reward += 50; 
    if (createdLineOfThree(board, lastAction, currentPlayer)) reward += 10; 

    return reward;
}

// FONCTIONS STANDARDS ET UTILITAIRES
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

function didBlockOpponentWin(g, col, opponent) {
    let r = 0;
    while (r < ROWS && g[r][col] === 0) r++;
    if (r === ROWS) return false; 

    const originalColor = g[r][col];
    g[r][col] = opponent; 
    const blocked = checkWinner(g, opponent);
    g[r][col] = originalColor; 
    
    return blocked;
}

function createdLineOfThree(g, col, player) {
    let r = 0;
    while (r < ROWS && g[r][col] === 0) r++;
    if (r === ROWS) return false;

    let linesOfThree = 0;

    const checkWindow = (r1, c1, r2, c2, r3, c3, r4, c4) => {
        if (r1<0 || r1>=ROWS || c1<0 || c1>=COLS ||
            r4<0 || r4>=ROWS || c4<0 || c4>=COLS) return;

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

// BINDING DES BOUTONS
document.getElementById('btn-reset').onclick = () => { 
    isArena = false; stopTrainingRequested = true; 
    lastAITurnContext = null;
    initBoard(); renderBoard(); 
    document.getElementById('status').innerText = "À toi vs IA-A."; 
};

function toggleTraining(aiName, btnId, originalText) {
    const btn = document.getElementById(btnId);
    if (isTraining && !stopTrainingRequested) {
        stopTrainingRequested = true;
        btn.innerText = "Arrêt en cours...";
    } else if (!isTraining) {
        btn.innerText = `🛑 Stopper l'IA-${aiName}`;
        runTraining(aiName).then(() => {
            btn.innerText = originalText; 
        });
    }
}

document.getElementById('btn-train-a').onclick = () => toggleTraining('A', 'btn-train-a', "Entraîner IA-A");
document.getElementById('btn-train-b').onclick = () => toggleTraining('B', 'btn-train-b', "Entraîner IA-B");
document.getElementById('btn-arena').onclick = () => runArena();

// Lancement au chargement
initBoard(); 
initIA();
