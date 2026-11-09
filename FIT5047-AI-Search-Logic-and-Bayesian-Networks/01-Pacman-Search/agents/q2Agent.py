# agents/q2Agent.py
# Compact, high-performance Alpha-Beta Pacman agent.
# Features: iterative deepening, adaptive time control, maze-distance cache,
# mode switching (EAT/CAP/HUNT), quiescence, history heuristic, aspiration
# windows, bounded transposition table (bounds+exact), loop/oscillation shaping,
# ghost-aware move ordering. Always returns a legal action.

import math, time
from typing import List, Tuple, Optional, Dict
from collections import deque, defaultdict

import util
from game import Agent, Directions
from pacman import GameState
from util import manhattanDistance

try:
    from logs.search_logger import log_function
except Exception:
    def log_function(f):
        return f

# ------------------------------- config ---------------------------------- #
CFG = {
    "QUIESCENCE_EXTRA_ON_VOLATILE": 1,
    "ORDER_STOP_PEN": 0.70,
    "ORDER_REVERSE_PEN": 0.14,
    "ORDER_FOOD_REWARD": 1.50,
    "ORDER_CAP_REWARD": 2.60,
    "ORDER_FLEE_REWARD": 0.42,
    "ORDER_APPROACH_PEN": 0.52,
    "ORDER_JUNCTION_REWARD": 0.35,
    "ORDER_LOOP_PEN": 0.90,
    "ORDER_KILLER_BONUS": 0.15,
    "ASP_INIT": 80.0,
    "ASP_GROW": 2.0,
    "ASP_MAX": 5000.0,
    "HIST_BONUS": 8.0,
}

# --------------------------- maze-distance (BFS) -------------------------- #
_MD_CACHE: Dict[Tuple[int, Tuple[int,int], Tuple[int,int]], int] = {}
_MD_MAX = 15000

def _walls_id(walls) -> int:
    return id(getattr(walls, "data", walls))

def _maze_dist(pos: Tuple[int,int], tgt: Tuple[int,int], walls) -> int:
    if pos == tgt: return 0
    key = (_walls_id(walls), pos, tgt)
    d = _MD_CACHE.get(key)
    if d is not None: return d
    W, H = walls.width, walls.height
    q, seen = deque([(pos,0)]), {pos}
    while q:
        (x,y), dist = q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0<=nx<W and 0<=ny<H and not walls[nx][ny] and (nx,ny) not in seen:
                nd = dist+1
                if (nx,ny)==tgt:
                    if len(_MD_CACHE)>_MD_MAX: _MD_CACHE.clear()
                    _MD_CACHE[key]=nd
                    return nd
                seen.add((nx,ny))
                q.append(((nx,ny),nd))
    if len(_MD_CACHE)>_MD_MAX: _MD_CACHE.clear()
    _MD_CACHE[key]=10**6
    return 10**6

def _grid_degree(walls, pos: Tuple[int,int]) -> int:
    x,y = pos; W,H = walls.width, walls.height; deg=0
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx,ny=x+dx,y+dy
        if 0<=nx<W and 0<=ny<H and not walls[nx][ny]: deg+=1
    return deg

def _near_live_ghost_maze_dist(state: GameState, pos, walls) -> int:
    best=99
    for g in state.getGhostStates():
        if g.scaredTimer>2: continue
        gx,gy=g.getPosition(); gm=manhattanDistance(pos,(int(gx),int(gy)))
        if gm<=5:
            d=_maze_dist(pos,(int(gx),int(gy)),walls)
            if d<best: best=d
    return best

# -------------- safety helpers -------------- #
def _live_ghost_positions(state: GameState):
    return [(int(g.getPosition()[0]), int(g.getPosition()[1])) for g in state.getGhostStates() if g.scaredTimer<=2]

def _min_maze_to_targets(pos: Tuple[int,int], tgts: List[Tuple[int,int]], walls) -> int:
    if not tgts: return 99
    md=99
    for t in tgts:
        md=min(md,_maze_dist(pos,t,walls))
    return md if md<10**6 else 99

# --------------------------------- modes ---------------------------------- #
_MODE_EAT, _MODE_CAP, _MODE_HUNT = 0,1,2

def _select_mode(state: GameState) -> int:
    pos=state.getPacmanPosition(); walls=state.getWalls(); caps=state.getCapsules()
    if max((g.scaredTimer for g in state.getGhostStates()), default=0)>=6:
        return _MODE_HUNT
    min_live_manh=min((manhattanDistance(pos,(int(g.getPosition()[0]),int(g.getPosition()[1])))
                       for g in state.getGhostStates() if g.scaredTimer<=2), default=99)
    min_live_maze=99
    if min_live_manh<=6:
        for g in state.getGhostStates():
            if g.scaredTimer<=2:
                gx,gy=g.getPosition();
                min_live_maze=min(min_live_maze,_maze_dist(pos,(int(gx),int(gy)),walls))
    if caps:
        cap_steps=min((_maze_dist(pos,c,walls) for c in caps), default=99)
        if (min_live_maze<=4) or (min_live_manh<=3 and cap_steps<=8) or (cap_steps<=3):
            return _MODE_CAP
    return _MODE_EAT

# -------------------------------- evaluation ------------------------------ #

def _ghost_features(state: GameState, pos) -> Tuple[float,float,bool,int]:
    danger=0.0; chase=0.0; threat=False; min_live=9999
    for g in state.getGhostStates():
        gx,gy=g.getPosition(); d=manhattanDistance(pos,(int(gx),int(gy)))
        if g.scaredTimer>2:
            chase+=7.0/(d+1.0)
        else:
            if d==0: return (1e6,0.0,True,0)
            if d<=1: danger+=1050.0
            elif d==2: danger+=420.0
            else: danger+=22.0/(d+0.2)
            if d<min_live: min_live=d
            if d<=4: threat=True
    if min_live==9999: min_live=99
    return danger,chase,threat,min_live

def _food_pull(pos, foodList: List[Tuple[int,int]], walls) -> Tuple[float,int]:
    if not foodList: return 0.0,0
    shortlist=sorted(foodList,key=lambda f:manhattanDistance(pos,f))[:4]
    dists=[_maze_dist(pos,f,walls) for f in shortlist]
    dists=[d for d in dists if d<10**6]
    if not dists: return 0.0,0
    return sum(1.0/(d+1.0) for d in dists), min(dists)

def betterEvaluationFunction(state: GameState) -> float:
    pos=state.getPacmanPosition(); food=state.getFood().asList(); caps=state.getCapsules()
    walls=state.getWalls(); base=state.getScore(); mode=_select_mode(state)
    food_pull,min_food=_food_pull(pos,food,walls); num_food=len(food)
    num_caps=len(caps); cap_d=min((_maze_dist(pos,c,walls) for c in caps), default=0) if num_caps else 0
    danger,chase,threat,min_live=_ghost_features(state,pos)
    near=_near_live_ghost_maze_dist(state,pos,walls)
    if near<=3: danger+=(4-near)*60.0
    elif near<=5: danger+=(6-near)*22.0
    deg=_grid_degree(walls,pos); corridor=0.0
    if deg<=1 and min_live<=5: corridor=(6-min_live)*22.0
    elif deg==2 and min_live<=3: corridor=20.0
    end_bonus=0.0
    if num_food<=5:
        end_bonus+=(6-num_food)*35.0
        end_bonus+=12.0/(min_food+1.0 if min_food>0 else 1.0)
    cap_pull=((20.0 if threat else 6.0)/(cap_d+1.0)) if num_caps else 0.0
    if mode==_MODE_HUNT: wf, wc, wm, wch, wcp = 28.0,-2.6,-0.6,25.0,4.0
    elif mode==_MODE_CAP: wf, wc, wm, wch, wcp = 20.0,-2.8,-0.6,14.0,26.0
    else: wf, wc, wm, wch, wcp = 42.0,-3.2,-1.0,18.0,8.0
    v = (1.0*base + wf*food_pull + wc*num_food + wm*min_food + wch*chase + wcp*cap_pull + end_bonus - danger - corridor)
    if state.isWin(): v+=1e6
    elif state.isLose(): v-=1e6
    return float(v)

# --------------------------------- agent ---------------------------------- #
_REVERSE={Directions.NORTH:Directions.SOUTH,Directions.SOUTH:Directions.NORTH,
          Directions.EAST:Directions.WEST,Directions.WEST:Directions.EAST,
          Directions.STOP:Directions.STOP}

class Q2_Agent(Agent):
    def __init__(self, evalFn='betterEvaluationFunction', maxDepth='5', perMoveMs='60', quiescence='1', gameBudget='26.5', **kw):
        if 'depth' in kw: maxDepth=kw['depth']
        if 'maxdepth' in kw: maxDepth=kw['maxdepth']
        if 'evalFn' in kw: evalFn=kw['evalFn']
        if 'perMove' in kw: perMoveMs=kw['perMove']
        if 'move_ms' in kw: perMoveMs=kw['move_ms']
        if 'gameBudgetSec' in kw: gameBudget=kw['gameBudgetSec']
        self.index=0
        try: self.maxDepth=int(maxDepth)
        except: self.maxDepth=5
        try: self.perMoveBudget=max(5,int(perMoveMs))/1000.0
        except: self.perMoveBudget=0.06
        try: self.quiescence=max(0,int(quiescence))
        except: self.quiescence=1
        try: self.gameBudget=float(gameBudget)
        except: self.gameBudget=26.5
        self.evaluationFunction=util.lookup(evalFn, globals())
        self._t_game_start=None
        self._eval_cache: Dict[Tuple,float]={}
        # TT: key -> (depth, lb, ub, exact)
        self._tt: Dict[Tuple,Tuple[int,float,float,Optional[float]]]={}
        self._last_best=None
        self._recent: List[Tuple[int,int]]=[]
        self._mode_now=_MODE_EAT
        self._hist=defaultdict(float)
        self._last_depth_time=None
        self._branch_factor_est=2.0

    # ------------- helpers ------------- #
    def _now(self)->float: return time.perf_counter()
    def _game_time_left(self)->float:
        if self._t_game_start is None: return self.gameBudget
        return max(0.0, self.gameBudget-(self._now()-self._t_game_start))
    def _state_key(self,state:GameState)->Tuple:
        p=state.getPacmanPosition()
        ghosts=tuple((int(g.getPosition()[0]),int(g.getPosition()[1]),int(bool(g.scaredTimer))) for g in state.getGhostStates())
        return (p,state.getNumFood(),len(state.getCapsules()),ghosts)
    def _eval(self,state:GameState)->float:
        k=self._state_key(state); v=self._eval_cache.get(k)
        if v is not None: return v
        v=float(self.evaluationFunction(state)); self._eval_cache[k]=v; return v
    def _is_volatile(self,state:GameState)->bool:
        pos=state.getPacmanPosition(); walls=state.getWalls(); danger,_,threat,min_live=_ghost_features(state,pos)
        if danger>=420.0 or threat or min_live<=2: return True
        caps=state.getCapsules()
        if caps and min((_maze_dist(pos,c,walls) for c in caps), default=99)<=2: return True
        return False
    def _min_live_ghost_dist(self,state:GameState)->int:
        _,_,_,m=_ghost_features(state,state.getPacmanPosition()); return m

    # ------------- ordering ------------- #
    def _order_pac_actions(self,state:GameState,actions:List[str])->List[str]:
        if not actions: return actions
        mode=self._mode_now; curr_dir=state.getPacmanState().getDirection()
        food=set(state.getFood().asList()); caps=list(state.getCapsules()); caps_set=set(caps)
        walls=state.getWalls(); pos_now=state.getPacmanPosition()
        g_before=self._min_live_ghost_dist(state)
        near_before=_near_live_ghost_maze_dist(state,pos_now,walls)
        cap_steps_now=None
        if mode==_MODE_CAP and caps:
            cap_steps_now=min((_maze_dist(pos_now,c,walls) for c in caps), default=None)
        threat=any((g.scaredTimer<=2 and manhattanDistance(pos_now,(int(g.getPosition()[0]),int(g.getPosition()[1])))<=4) for g in state.getGhostStates())
        last2=self._recent[-2:] if len(self._recent)>=2 else []
        scored=[]
        for a in actions:
            succ=state.generateSuccessor(0,a); s=self._eval(succ); pos=succ.getPacmanPosition()
            if a==Directions.STOP: s-=CFG["ORDER_STOP_PEN"]
            if a==_REVERSE.get(curr_dir,Directions.STOP): s-=(0.08 if mode==_MODE_CAP else CFG["ORDER_REVERSE_PEN"])
            if pos in food: s+=CFG["ORDER_FOOD_REWARD"]
            if pos in caps_set: s+=CFG["ORDER_CAP_REWARD"]
            g_after=self._min_live_ghost_dist(succ); delta=g_after-g_before
            if delta>0: s+=CFG["ORDER_FLEE_REWARD"]*min(delta,3)
            elif delta<0: s+=CFG["ORDER_APPROACH_PEN"]*delta
            near_after=_near_live_ghost_maze_dist(succ,pos,walls); nd=near_after-near_before
            if nd>0: s+=0.55*min(nd,3)
            elif nd<0: s+=0.12*nd
            if mode==_MODE_CAP and caps and cap_steps_now is not None:
                cap_steps_next=min((_maze_dist(pos,c,walls) for c in caps), default=None)
                if cap_steps_next is not None and cap_steps_next<cap_steps_now:
                    s+=0.45*(cap_steps_now-cap_steps_next)
            if threat and _grid_degree(walls,pos)>=3: s+=CFG["ORDER_JUNCTION_REWARD"]
            if pos in self._recent: s-=CFG["ORDER_LOOP_PEN"]
            if len(last2)==2 and pos==last2[0] and state.getPacmanPosition()==last2[1]: s-=CFG["ORDER_LOOP_PEN"]*1.25
            if self._last_best and a==self._last_best: s+=CFG["ORDER_KILLER_BONUS"]
            s+=0.01*self._hist[a]
            scored.append((s,a))
        scored.sort(reverse=True)
        return [a for _,a in scored]

    def _order_ghost_actions(self,state:GameState,agent:int,actions:List[str])->List[str]:
        if not actions: return actions
        scored=[(self._eval(state.generateSuccessor(agent,a)),a) for a in actions]
        scored.sort()
        return [a for _,a in scored]

    # ------------- panic ------------- #
    def _panic_action(self,state: GameState)->str:
        legal=state.getLegalActions(0)
        actions=[a for a in legal if a!=Directions.STOP] or legal
        if not actions: return Directions.STOP
        walls=state.getWalls(); caps=set(state.getCapsules()); live=_live_ghost_positions(state)
        scored=[]
        for a in actions:
            succ=state.generateSuccessor(0,a); pos=succ.getPacmanPosition()
            d=_min_maze_to_targets(pos,live,walls)
            m=min((manhattanDistance(pos,g) for g in live), default=99)
            deg=_grid_degree(walls,pos)
            s=5.0*d+0.6*deg+0.4*m
            if pos in caps: s+=6.0
            if m<=1: s-=100.0
            scored.append((s,a))
        scored.sort(reverse=True)
        return scored[0][1]

    # ------------- TT helpers ------------- #
    def _tt_get(self,key:Tuple,dleft:int,alpha:float,beta:float):
        entry=self._tt.get(key)
        if not entry: return None
        depth,lb,ub,exact=entry
        if depth<dleft: return None
        if exact is not None: return exact
        if lb is not None and lb>=beta: return lb
        if ub is not None and ub<=alpha: return ub
        return None
    def _tt_put_exact(self,key:Tuple,dleft:int,score:float): self._tt[key]=(dleft,-math.inf,math.inf,score)
    def _tt_put_lower(self,key:Tuple,dleft:int,score:float):
        old=self._tt.get(key)
        if old and old[0]>dleft: return
        self._tt[key]=(dleft,score,math.inf,None)
    def _tt_put_upper(self,key:Tuple,dleft:int,score:float):
        old=self._tt.get(key)
        if old and old[0]>dleft: return
        self._tt[key]=(dleft,-math.inf,score,None)

    # ------------- search ------------- #
    def _alphabeta(self,state:GameState,dleft:int,agent:int,alpha:float,beta:float,deadline:float)->Tuple[float,Optional[str]]:
        if self._now()>=deadline: raise TimeoutError
        if state.isWin() or state.isLose() or dleft==0:
            ext=self.quiescence+(CFG["QUIESCENCE_EXTRA_ON_VOLATILE"] if self._is_volatile(state) else 0)
            cur=state; score=self._eval(cur)
            while ext>0 and self._is_volatile(cur):
                if self._now()>=deadline: break
                pacts=cur.getLegalActions(0); pacts=[a for a in pacts if a!=Directions.STOP] or pacts
                if not pacts: break
                best=max((self._eval(cur.generateSuccessor(0,a)),a) for a in pacts)[1]
                cur=cur.generateSuccessor(0,best)
                for g in range(1,cur.getNumAgents()):
                    gacts=cur.getLegalActions(g) or []
                    if not gacts: continue
                    worst=min((self._eval(cur.generateSuccessor(g,a)),a) for a in gacts)[1]
                    cur=cur.generateSuccessor(g,worst)
                score=self._eval(cur); ext-=1
            return score,None
        numAgents=state.getNumAgents(); key=(agent,dleft,self._state_key(state))
        tt=self._tt_get(key,dleft,alpha,beta)
        if tt is not None: return tt,None
        if agent==0:
            legal=state.getLegalActions(0); actions=[a for a in legal if a!=Directions.STOP] or legal
            if not actions:
                v=self._eval(state); self._tt_put_exact(key,dleft,v); return v,Directions.STOP
            best_v=-math.inf; best_a=Directions.STOP
            for a in self._order_pac_actions(state,actions):
                succ=state.generateSuccessor(0,a)
                v,_=self._alphabeta(succ,dleft,1,alpha,beta,deadline)
                if v>best_v or (v==best_v and best_a==Directions.STOP and a!=Directions.STOP): best_v,best_a=v,a
                if best_v>=beta:
                    self._hist[a]+=CFG["HIST_BONUS"]; self._tt_put_lower(key,dleft,best_v); return best_v,best_a
                alpha=max(alpha,best_v)
            self._tt_put_exact(key,dleft, best_v); return best_v,best_a
        else:
            legal=state.getLegalActions(agent) or []
            if not legal:
                v=self._eval(state); self._tt_put_exact(key,dleft,v); return v,None
            nextAgent=(agent+1)%numAgents; nextDepth=dleft-1 if nextAgent==0 else dleft
            worst=math.inf
            for a in self._order_ghost_actions(state,agent,legal):
                succ=state.generateSuccessor(agent,a)
                v,_=self._alphabeta(succ,nextDepth,nextAgent,alpha,beta,deadline)
                if v<worst: worst=v
                if worst<=alpha:
                    self._tt_put_upper(key,dleft,worst); return worst,None
                beta=min(beta,worst)
            self._tt_put_exact(key,dleft,worst); return worst,None

    # ------------- time mgmt ------------- #
    def _compute_move_budget(self,state:GameState)->float:
        base=self.perMoveBudget; left=self._game_time_left()
        if left<=0.006: return 0.004
        boost=0.0
        if self._is_volatile(state): boost+=0.015
        elif state.getNumFood()<=8: boost+=0.007
        m=_select_mode(state)
        if m in (_MODE_CAP,_MODE_HUNT): boost+=0.01
        return min(base+boost, max(0.004, left*0.03))

    # ------------- action ------------- #
    @log_function
    def getAction(self, gameState: GameState) -> str:
        if self._t_game_start is None: self._t_game_start=self._now()
        budget=self._compute_move_budget(gameState); deadline=self._now()+budget
        self._eval_cache.clear(); self._tt.clear(); self._mode_now=_select_mode(gameState)
        cur_pos=gameState.getPacmanPosition(); self._recent.append(cur_pos)
        if len(self._recent)>12: self._recent.pop(0)
        near0=_near_live_ghost_maze_dist(gameState,cur_pos,gameState.getWalls())
        if self._min_live_ghost_dist(gameState)<=1 or near0<=2:
            a=self._panic_action(gameState)
            self._last_best=a
            return a
        legal0=gameState.getLegalActions(0); legal0=[a for a in legal0 if a!=Directions.STOP] or legal0
        if not legal0: return Directions.STOP
        greedy=max(legal0, key=lambda a:self._eval(gameState.generateSuccessor(0,a)))
        best_action=greedy
        try:
            d=1; last_value=self._eval(gameState); asp=CFG["ASP_INIT"]
            while d<=self.maxDepth:
                if self._last_depth_time is not None:
                    eta=self._last_depth_time*(self._branch_factor_est)
                    if self._now()+eta>deadline: break
                alpha_c=last_value-asp; beta_c=last_value+asp
                t0=self._now(); val,act=self._alphabeta(gameState,d,0,alpha_c,beta_c,deadline)
                self._last_depth_time=self._now()-t0
                self._branch_factor_est=max(1.6, min(3.2, self._branch_factor_est*0.7+2.2*0.3))
                if (alpha_c<val<beta_c) or asp>=CFG["ASP_MAX"]:
                    last_value=val; best_action=act or best_action; d+=1; asp=CFG["ASP_INIT"]
                else:
                    asp*=CFG["ASP_GROW"]
                    if self._now()>=deadline: break
                    alpha_c=last_value-asp; beta_c=last_value+asp
                    val2,act2=self._alphabeta(gameState,d,0,alpha_c,beta_c,deadline)
                    last_value=val2; best_action=act2 or best_action; d+=1; asp=CFG["ASP_INIT"]
                if last_value>9e5 or gameState.isWin(): break
        except TimeoutError:
            pass
        except Exception:
            pass
        self._last_best=best_action
        if best_action not in legal0:
            best_action = greedy if greedy in legal0 else (legal0[0] if legal0 else Directions.STOP)
        return best_action
