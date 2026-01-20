let ws = null;
const logContainer = document.getElementById('system-logs');

// 角色名称映射
const ROLE_NAMES = {
    werewolf: '🐺 狼人',
    seer: '🔮 预言家',
    witch: '🧪 女巫',
    villager: '👨🌾 村民'
};

function logMessage(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;

    if (type === 'error') {
        logEntry.style.color = '#e74c3c';
    } else if (type === 'warning') {
        logEntry.style.color = '#f39c12';
    } else if (type === 'success') {
        logEntry.style.color = '#2ecc71';
    }

    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        logMessage('已经连接到服务器', 'warning');
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/observer`;

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
        logMessage('成功连接到WebSocket服务器', 'success');
        document.getElementById('connection-status').className = 'connection-status connected';
        document.getElementById('connection-status').textContent = '已连接';

        // 订阅频道
        ws.send(JSON.stringify({
            type: 'subscribe',
            channels: ['game_state', 'speech', 'game_events']
        }));
    };

    ws.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            logMessage('解析消息失败: ' + error, 'error');
        }
    };

    ws.onclose = function () {
        logMessage('WebSocket连接已关闭', 'warning');
        document.getElementById('connection-status').className = 'connection-status disconnected';
        document.getElementById('connection-status').textContent = '未连接';
    };

    ws.onerror = function (error) {
        logMessage('WebSocket错误: ' + error, 'error');
    };
}

function disconnectWebSocket() {
    if (ws) {
        ws.close();
        ws = null;
    }
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'game_state':
            updateGameState(message.data);
            break;
        case 'speech':
            addSpeechMessage(message.data);
            break;
        case 'game_event':
            logMessage(`事件: ${message.data.description}`, 'info');
            break;
        case 'system_alert':
            logMessage(`系统: ${message.data.message}`, 'warning');
            break;
    }
}

function updateGameState(state) {
    // 更新游戏信息
    document.getElementById('game-phase').textContent = state.phase;
    document.getElementById('day-number').textContent = state.day_number;
    document.getElementById('alive-count').textContent = state.alive_players.length;

    // 更新玩家网格
    const playersGrid = document.getElementById('players-grid');
    playersGrid.innerHTML = '';

    // 这里应该根据实际数据更新玩家卡片
    // 简化示例
    const samplePlayers = [
        {id: 'player1', name: '玩家1', role: 'werewolf', alive: true},
        {id: 'player2', name: '玩家2', role: 'seer', alive: true},
        {id: 'player3', name: '玩家3', role: 'villager', alive: true},
        {id: 'player4', name: '玩家4', role: 'werewolf', alive: true},
        {id: 'player5', name: '玩家5', role: 'witch', alive: true},
        {id: 'player6', name: '玩家6', role: 'villager', alive: true},
    ];

    samplePlayers.forEach(player => {
        const card = document.createElement('div');
        card.className = `player-card ${player.role}`;
        card.id = `player-${player.id}`;

        card.innerHTML = `
            <div class="player-header">
                <div class="player-name">${player.name}</div>
                <div class="player-role">${ROLE_NAMES[player.role]}</div>
            </div>
            <div class="player-stats">
                <div class="stat-item">
                    <span>状态:</span>
                    <span>${player.alive ? '存活' : '死亡'}</span>
                </div>
                <div class="stat-item">
                    <span>怀疑指数:</span>
                    <span>65%</span>
                </div>
                <div class="stat-item">
                    <span>发言次数:</span>
                    <span>3</span>
                </div>
            </div>
        `;

        playersGrid.appendChild(card);
    });
}

function addSpeechMessage(speech) {
    const speechContainer = document.getElementById('speech-container');
    const bubble = document.createElement('div');
    bubble.className = 'speech-bubble';

    const time = new Date(speech.timestamp * 1000).toLocaleTimeString();
    bubble.innerHTML = `
        <strong>${speech.player_name} [${time}]:</strong><br>
        ${speech.text}
    `;

    speechContainer.appendChild(bubble);
    speechContainer.scrollTop = speechContainer.scrollHeight;

    // 限制最多显示10条发言
    const bubbles = speechContainer.getElementsByClassName('speech-bubble');
    if (bubbles.length > 10) {
        speechContainer.removeChild(bubbles[0]);
    }
}

function requestGameState() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'request_state'
        }));
        logMessage('已请求游戏状态', 'info');
    }
}

function revealAllRoles() {
    logMessage('显示所有玩家身份 (管理员功能)', 'warning');
    // 这里可以添加显示所有角色的逻辑
}

function exportGameData() {
    logMessage('导出游戏数据功能', 'info');
    // 这里可以添加导出数据的逻辑
}

function calculateAnalysis() {
    logMessage('重新计算分析指标', 'info');
    // 这里可以添加分析计算的逻辑
}

function showVotePatterns() {
    logMessage('显示投票模式分析', 'info');
    // 这里可以添加投票分析的逻辑
}

function detectAlliances() {
    logMessage('检测玩家联盟关系', 'info');
    // 这里可以添加联盟检测的逻辑
}

// 页面加载时自动连接
window.onload = function () {
    logMessage('观察者界面已加载', 'info');
    // 自动连接可以在这里启用
    // connectWebSocket();
};