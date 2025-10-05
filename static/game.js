// 遊戲設定
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const TILE_COUNT = 20;
const TILE_SIZE = canvas.width / TILE_COUNT;

let snake = [];
let velocity = { x: 1, y: 0 };
let food = { x: 10, y: 10 };
let gameInterval = null;
let speed = 120;
let running = false;
let score = 0;
let aiEnabled = false;

// 初始化遊戲
function initGame() {
    snake = [
        { x: 9, y: 10 },
        { x: 8, y: 10 },
        { x: 7, y: 10 }
    ];
    velocity = { x: 1, y: 0 };
    score = 0;
    placeFood();
    updateStats();
    draw();
}

// 放置食物
function placeFood() {
    do {
        food.x = Math.floor(Math.random() * TILE_COUNT);
        food.y = Math.floor(Math.random() * TILE_COUNT);
    } while (snake.some(s => s.x === food.x && s.y === food.y));
}

// 繪製
function draw() {
    // 清空畫布
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 繪製網格
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    for (let i = 0; i <= TILE_COUNT; i++) {
        ctx.beginPath();
        ctx.moveTo(i * TILE_SIZE, 0);
        ctx.lineTo(i * TILE_SIZE, canvas.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * TILE_SIZE);
        ctx.lineTo(canvas.width, i * TILE_SIZE);
        ctx.stroke();
    }

    // 繪製食物
    ctx.fillStyle = '#ff4444';
    ctx.fillRect(food.x * TILE_SIZE + 2, food.y * TILE_SIZE + 2, TILE_SIZE - 4, TILE_SIZE - 4);

    // 繪製蛇
    snake.forEach((segment, index) => {
        if (index === 0) {
            ctx.fillStyle = '#44ff44'; // 蛇頭
        } else {
            ctx.fillStyle = '#00cc00'; // 蛇身
        }
        ctx.fillRect(segment.x * TILE_SIZE + 2, segment.y * TILE_SIZE + 2, TILE_SIZE - 4, TILE_SIZE - 4);
    });
}

// 移動蛇
function moveSnake() {
    if (!running) return;

    const head = {
        x: (snake[0].x + velocity.x + TILE_COUNT) % TILE_COUNT,
        y: (snake[0].y + velocity.y + TILE_COUNT) % TILE_COUNT
    };

    // 檢查是否撞到自己
    if (snake.some(s => s.x === head.x && s.y === head.y)) {
        gameOver();
        return;
    }

    snake.unshift(head);

    // 檢查是否吃到食物
    if (head.x === food.x && head.y === food.y) {
        score += 10;
        updateStats();
        placeFood();
    } else {
        snake.pop();
    }

    draw();
}

// 更新統計
function updateStats() {
    document.getElementById('score').textContent = score;
    document.getElementById('length').textContent = snake.length;
}

// 遊戲結束
function gameOver() {
    running = false;
    if (gameInterval) {
        clearInterval(gameInterval);
        gameInterval = null;
    }

    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 30px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('遊戲結束!', canvas.width / 2, canvas.height / 2 - 20);
    ctx.font = '20px sans-serif';
    ctx.fillText(`分數: ${score}`, canvas.width / 2, canvas.height / 2 + 20);
    ctx.font = '16px sans-serif';
    ctx.fillText('按 R 或點擊重新開始', canvas.width / 2, canvas.height / 2 + 50);
}

// 開始遊戲
function startGame() {
    if (running) return;
    running = true;
    gameInterval = setInterval(moveSnake, speed);
}

// 重新開始
function resetGame() {
    if (gameInterval) {
        clearInterval(gameInterval);
        gameInterval = null;
    }
    running = false;
    initGame();
    startGame();
}

// 設定方向
function setDirection(dir) {
    const directions = {
        'up': { x: 0, y: -1 },
        'down': { x: 0, y: 1 },
        'left': { x: -1, y: 0 },
        'right': { x: 1, y: 0 }
    };

    const newVel = directions[dir.toLowerCase()];
    if (!newVel) return;

    // 防止 180 度轉向
    if (newVel.x === -velocity.x && newVel.y === -velocity.y) return;

    velocity = newVel;
}

// 鍵盤控制
document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    
    if (key === 'r') {
        resetGame();
        return;
    }

    if (!aiEnabled) {
        if (['arrowup', 'w'].includes(key)) setDirection('up');
        if (['arrowdown', 's'].includes(key)) setDirection('down');
        if (['arrowleft', 'a'].includes(key)) setDirection('left');
        if (['arrowright', 'd'].includes(key)) setDirection('right');
    }
});

// 切換 AI 模式
function toggleAI() {
    aiEnabled = !aiEnabled;
    const text = document.getElementById('aiToggleText');
    const badge = document.getElementById('cameraBadge');
    
    if (aiEnabled) {
        text.textContent = '🔴 Disable AI';
        if (badge) {
            badge.className = 'badge active';
            badge.textContent = 'Active';
        }
        startPredictionPolling();
    } else {
        text.textContent = '🤖 Enable AI';
        if (badge) {
            badge.className = 'badge';
            badge.textContent = 'Inactive';
        }
        stopPredictionPolling();
    }
}

// 輪詢預測
let pollInterval = null;

function startPredictionPolling() {
    if (pollInterval) return;
    
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch('/prediction');
            const data = await response.json();
            
            if (data.predictions) {
                updatePredictions(data.predictions);
            }
            
            if (data.direction && aiEnabled) {
                setDirection(data.direction);
                document.getElementById('currentDirection').textContent = data.direction;
            }
        } catch (error) {
            console.error('預測錯誤:', error);
        }
    }, 120);
}

function stopPredictionPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// 更新預測顯示
function updatePredictions(predictions) {
    const labels = ['up', 'left', 'right', 'down'];
    
    predictions.forEach((prob, index) => {
        const label = labels[index];
        const percentage = (prob * 100).toFixed(1);
        
        const fill = document.getElementById(`pred-${label}`);
        const value = document.getElementById(`val-${label}`);
        
        if (fill && value) {
            fill.style.width = `${prob * 100}%`;
            value.textContent = `${percentage}%`;
        }
    });
}

// 初始化
initGame();
startGame();
