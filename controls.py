# controls.py
# 四個控制函式獨立在此模組，供 controller.py 呼叫，以及 game.py 讀取當前方向

import threading

_lock = threading.Lock()
_current_direction = 'right'  # 初始方向

# Prevent immediate 180 degree turns: define opposites
_opposites = {
    'up': 'down',
    'down': 'up',
    'left': 'right',
    'right': 'left'
}

def _set_direction(dir_name):
    global _current_direction
    with _lock:
        if dir_name not in _opposites:
            return
        # 禁止 180 度直接轉向
        if _opposites[dir_name] == _current_direction:
            return
        _current_direction = dir_name

def up():
    _set_direction('up')

def down():
    _set_direction('down')

def left():
    _set_direction('left')

def right():
    _set_direction('right')

def get_direction():
    with _lock:
        return _current_direction

# 若需要外部重設方向
def reset():
    global _current_direction
    with _lock:
        _current_direction = 'right'
