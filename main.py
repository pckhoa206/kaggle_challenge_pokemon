import os
import random

from cg.api import (
    Observation,
    to_observation_class,
    OptionType,
    SelectType,
    SelectContext,
    AreaType,
    CardType,
    all_card_data,
    all_attack,
    search_begin,
    search_step,
    search_end,
    search_release
)

# Global card and attack database maps for Heuristic logic
try:
    CARD_DATA_MAP = {c.cardId: c for c in all_card_data()}
except Exception:
    CARD_DATA_MAP = {}

try:
    ATTACK_DMG_MAP = {a.attackId: a.damage for a in all_attack()}
except Exception:
    ATTACK_DMG_MAP = {}

try:
    ATTACK_MAP = {a.attackId: a for a in all_attack()}
except Exception:
    ATTACK_MAP = {}

def _read_deck_csv_top() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r") as file:
            csv = file.read().split("\n")
        return [int(line) for line in csv if line.strip()]
    except Exception:
        return []

try:
    _CURRENT_DECK = _read_deck_csv_top()
except Exception:
    _CURRENT_DECK = []

def is_playing_abomasnow() -> bool:
    return 723 in _CURRENT_DECK

OPPONENT_DECKS = {
    "aggro_rush": [3]*43 + [721]*4 + [1126]*1 + [1152]*4 + [1227]*4 + [1235]*4,
    "control_disrupt": [3]*35 + [721]*2 + [722]*4 + [723]*4 + [1145]*4 + [1158]*1 + [1205]*2 + [1227]*4 + [1235]*4,
    "tank_boss": [2]*40 + [756]*20,
    "turbo_energy": [3]*35 + [722]*4 + [723]*4 + [1126]*1 + [1145]*4 + [1205]*4 + [1227]*4 + [1235]*4,
    "fast_aggro": [2]*32 + [31]*4 + [46]*4 + [76]*4 + [1145]*4 + [1152]*4 + [1227]*4 + [1235]*4,
    "mill_stall": [6]*32 + [25]*4 + [27]*4 + [28]*4 + [1145]*4 + [1152]*4 + [1227]*4 + [1235]*4,
    "tank_heal": [1]*32 + [33]*4 + [35]*4 + [47]*4 + [1145]*4 + [1152]*4 + [1227]*4 + [1235]*4
}

def detect_opponent_deck(opp_known: list[int]) -> list[int]:
    # Count card occurrences in opp_known
    counts = {}
    for cid in opp_known:
        counts[cid] = counts.get(cid, 0) + 1
        
    # Check for specific cards that uniquely identify an archetype
    if 756 in counts or 2 in counts:  # Mega Kangaskhan ex or Fire energy
        if 756 in counts:
            return list(OPPONENT_DECKS["tank_boss"])
        else:
            return list(OPPONENT_DECKS["fast_aggro"])
            
    if 25 in counts or 27 in counts or 28 in counts or 6 in counts:  # Pinsir, Iron Leaves, Poltchageist, Fighting energy
        return list(OPPONENT_DECKS["mill_stall"])
        
    if 33 in counts or 35 in counts or 47 in counts or 1 in counts:  # Froakie, Walking Wake, Totodile, Grass energy
        return list(OPPONENT_DECKS["tank_heal"])
        
    if 722 in counts or 723 in counts:  # Snover, Mega Abomasnow ex
        if 721 in counts:  # Kyogre
            return list(OPPONENT_DECKS["control_disrupt"])
        else:
            return list(OPPONENT_DECKS["turbo_energy"])
            
    if 721 in counts:  # Kyogre only
        return list(OPPONENT_DECKS["aggro_rush"])
        
    # Fallback to our own deck list if we can't identify the opponent
    return list(_CURRENT_DECK) if _CURRENT_DECK else [3] * 60


# Opponent Hand Tracking database
_OPPONENT_HAND_TRACKED = {} # maps serial -> cardId

def update_opponent_hand_tracking(obs: Observation, your_idx: int):
    global _OPPONENT_HAND_TRACKED
    
    # Reset tracking at the start of the match
    if not obs.current or obs.current.turn <= 1:
        _OPPONENT_HAND_TRACKED = {}
        return
        
    opp_idx = 1 - your_idx
    logs = obs.logs or []
    for log in logs:
        try:
            ltype = int(log.type)
        except Exception:
            continue
            
        # 1. Card moving to opponent hand (revealed)
        if ltype == 6: # MOVE_CARD
            if log.playerIndex == opp_idx and log.toArea == AreaType.HAND:
                if log.serial and log.cardId:
                    _OPPONENT_HAND_TRACKED[log.serial] = log.cardId
                    
        # 2. Card moving out of opponent hand
        elif ltype in (6, 7): # MOVE_CARD or MOVE_CARD_REVERSE
            if log.playerIndex == opp_idx and log.fromArea == AreaType.HAND:
                _OPPONENT_HAND_TRACKED.pop(log.serial, None)
                    
        # 3. Actions played from hand
        elif ltype in (10, 11, 12): # PLAY, ATTACH, EVOLVE
            if log.playerIndex == opp_idx:
                _OPPONENT_HAND_TRACKED.pop(log.serial, None)

# Global variables for ONNX inference session
_ONNX_SESSION = None
_ONNX_LOADED = False

def init_onnx_model():
    """Attempt to initialize the ONNX model for deep reinforcement learning inference."""
    global _ONNX_SESSION, _ONNX_LOADED
    if _ONNX_LOADED:
        return
        
    onnx_path = "model.onnx"
    if not os.path.exists(onnx_path):
        # Check Kaggle submission directory
        onnx_path = "/kaggle_simulations/agent/" + onnx_path
        
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            # Use CPU execution provider for CPU efficiency and safety on Kaggle
            _ONNX_SESSION = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            _ONNX_LOADED = True
            print("ONNX model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load ONNX model ({e}). Falling back to Heuristic Agent.")
            _ONNX_LOADED = False
    else:
        _ONNX_LOADED = False

def extract_state_for_onnx(obs: Observation) -> list:
    """Extract flat state vector of size 164 matching the DRL Env extraction."""
    state = [0.0] * 164
    if not obs or not obs.current:
        return state
        
    your_idx = obs.current.yourIndex
    player = obs.current.players[your_idx]
    opponent = obs.current.players[1 - your_idx]
    
    # Player 0 (Agent) features
    idx = 0
    # Active Pokemon
    active = player.active[0] if player.active else None
    state[idx] = active.hp / active.maxHp if active else 0.0; idx += 1
    state[idx] = len(active.energies) / 5.0 if active else 0.0; idx += 1
    state[idx] = active.id / 1500.0 if active else 0.0; idx += 1
    
    # Bench Pokemon (max 5)
    for i in range(5):
        pkmn = player.bench[i] if i < len(player.bench) else None
        state[idx] = pkmn.hp / pkmn.maxHp if pkmn else 0.0; idx += 1
        state[idx] = len(pkmn.energies) / 5.0 if pkmn else 0.0; idx += 1
        state[idx] = pkmn.id / 1500.0 if pkmn else 0.0; idx += 1
        
    # Hand & Deck counts
    state[idx] = len(player.hand or []) / 10.0; idx += 1
    state[idx] = player.deckCount / 60.0; idx += 1
    state[idx] = len(player.prize) / 6.0; idx += 1
    
    # Hand Cards (max 10)
    hand_cards = player.hand or []
    for i in range(10):
        card = hand_cards[i] if i < len(hand_cards) else None
        card_data = CARD_DATA_MAP.get(card.id) if card else None
        state[idx] = card.id / 1500.0 if card else 0.0; idx += 1
        state[idx] = int(card_data.cardType) / 7.0 if card_data else 0.0; idx += 1
    
    # Player 1 (Opponent) features
    opp_active = opponent.active[0] if opponent.active else None
    state[idx] = opp_active.hp / opp_active.maxHp if opp_active else 0.0; idx += 1
    state[idx] = len(opp_active.energies) / 5.0 if opp_active else 0.0; idx += 1
    state[idx] = opp_active.id / 1500.0 if opp_active else 0.0; idx += 1
    
    for i in range(5):
        pkmn = opponent.bench[i] if i < len(opponent.bench) else None
        state[idx] = pkmn.hp / pkmn.maxHp if pkmn else 0.0; idx += 1
        state[idx] = len(pkmn.energies) / 5.0 if pkmn else 0.0; idx += 1
        state[idx] = pkmn.id / 1500.0 if pkmn else 0.0; idx += 1
        
    state[idx] = opponent.handCount / 10.0; idx += 1
    state[idx] = opponent.deckCount / 60.0; idx += 1
    state[idx] = len(opponent.prize) / 6.0; idx += 1
    
    # Game Phase / Turn features
    state[idx] = obs.current.turn / 50.0; idx += 1
    state[idx] = obs.select.context / 50.0 if obs.select else 0.0; idx += 1
    
    # Options features
    options = obs.select.option if (obs.select and obs.select.option) else []
    for i in range(50):
        if i < len(options):
            opt = options[i]
            state[idx] = opt.type / 16.0; idx += 1
            state[idx] = (getattr(opt, "attackId", 0) or 0) / 1200.0; idx += 1
        else:
            idx += 2 # pad with zeros
            
    return state

def run_onnx_inference(obs: Observation) -> list[int]:
    """Execute ONNX model inference and return best action indices."""
    global _ONNX_SESSION
    import numpy as np
    
    state_vec = extract_state_for_onnx(obs)
    
    # Expand dims to batch size 1 (1, 164)
    input_data = np.expand_dims(np.array(state_vec, dtype=np.float32), axis=0)
    
    # Feed to ONNX session
    input_name = _ONNX_SESSION.get_inputs()[0].name
    output_name = _ONNX_SESSION.get_outputs()[0].name
    logits = _ONNX_SESSION.run([output_name], {input_name: input_data})[0][0] # shape (50,)
    
    options = obs.select.option
    max_count = obs.select.maxCount
    your_idx = obs.current.yourIndex
    
    # 1. Primary choice is chosen purely by the ONNX RL Model
    best_logit = -999999.0
    primary_choice = 0
    for i in range(len(options)):
        val = float(logits[i]) if i < len(logits) else -999999.0
        if val > best_logit:
            best_logit = val
            primary_choice = i
            
    choices = [primary_choice]
    
    # 2. Secondary choices are chosen by Heuristic (matching training env logic)
    if max_count > 1 and len(options) > 1:
        scored_options = []
        for i in range(len(options)):
            if i == primary_choice:
                continue
            try:
                score = score_option(obs, options[i], obs.select.context, your_idx)
            except Exception:
                score = 100.0
            scored_options.append((score, i))
            
        scored_options.sort(key=lambda x: x[0], reverse=True)
        for _, idx in scored_options[:max_count - 1]:
            choices.append(idx)
            
    return choices

def read_deck_csv() -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck

def get_pokemon_from_option(obs: Observation, opt, your_idx: int):
    player_idx = opt.playerIndex if getattr(opt, "playerIndex", None) is not None else your_idx
    player = obs.current.players[player_idx]
    
    # 1. Check inPlayArea and inPlayIndex
    in_play_area = getattr(opt, "inPlayArea", None)
    in_play_index = getattr(opt, "inPlayIndex", None)
    if in_play_area is not None:
        if in_play_area == AreaType.ACTIVE:
            return player.active[0] if player.active else None
        elif in_play_area == AreaType.BENCH:
            if in_play_index is not None and 0 <= in_play_index < len(player.bench):
                return player.bench[in_play_index]
                
    # 2. Check area and index (when the option itself represents a card in play)
    area = getattr(opt, "area", None)
    index = getattr(opt, "index", None)
    if area is not None:
        if area == AreaType.ACTIVE:
            return player.active[0] if player.active else None
        elif area == AreaType.BENCH:
            if index is not None and 0 <= index < len(player.bench):
                return player.bench[index]
                
    return None

def get_card_id(obs: Observation, opt, your_idx: int) -> int:
    # Check cardId directly
    card_id = getattr(opt, "cardId", None)
    if card_id is not None and card_id != 0:
        return card_id
        
    # Check card in area
    area = getattr(opt, "area", None)
    index = getattr(opt, "index", None)
    player_idx = opt.playerIndex if getattr(opt, "playerIndex", None) is not None else your_idx
    
    if area is not None and index is not None:
        player = obs.current.players[player_idx]
        if area == AreaType.HAND:
            if player.hand and 0 <= index < len(player.hand):
                return player.hand[index].id
        elif area == AreaType.LOOKING:
            if obs.current.looking and 0 <= index < len(obs.current.looking):
                card = obs.current.looking[index]
                return card.id if card else 0
        elif area == AreaType.DISCARD:
            if player.discard and 0 <= index < len(player.discard):
                return player.discard[index].id
        elif area == AreaType.DECK:
            if obs.select and obs.select.deck and 0 <= index < len(obs.select.deck):
                return obs.select.deck[index].id
        elif area == AreaType.ACTIVE:
            if player.active:
                return player.active[0].id
        elif area == AreaType.BENCH:
            if 0 <= index < len(player.bench):
                return player.bench[index].id
                
    return 0

def get_max_attack_damage(obs, your_idx: int) -> int:
    """Calculate maximum damage our active Pokemon can deal."""
    player = obs.current.players[your_idx]
    active_pkmn = player.active[0] if player.active else None
    if not active_pkmn:
        return 0
        
    card_data = CARD_DATA_MAP.get(active_pkmn.id)
    if not card_data or not card_data.attacks:
        return 10
        
    max_dmg = 0
    for attack_id in card_data.attacks:
        a = ATTACK_MAP.get(attack_id)
        if a:
            dmg = a.damage
            if a.name == "Hammer-lanche":
                dmg = 300
            elif a.name == "Riptide":
                discard_pile = player.discard or []
                water_in_discard = sum(1 for c in discard_pile if c.id == 3)
                dmg = water_in_discard * 20
            elif a.name == "Crystal Fall":
                water_in_play = 0
                if player.active and player.active[0]:
                    water_in_play += len(player.active[0].energies)
                for pk in player.bench:
                    if pk:
                        water_in_play += len(pk.energies)
                dmg = 120 if water_in_play >= 4 else 30
            elif a.name == "Gale Thrust":
                dmg = 120
        else:
            dmg = ATTACK_DMG_MAP.get(attack_id, 0)
            
        if dmg > max_dmg:
            max_dmg = dmg
            
    return max_dmg if max_dmg > 0 else 10

def calculate_enemy_max_damage_next_turn(obs: Observation, opp_idx: int) -> int:
    """Safely estimate the maximum attack damage of the opponent active Pokemon next turn."""
    opponent = obs.current.players[opp_idx]
    opp_active = opponent.active[0] if opponent.active else None
    if not opp_active:
        return 0
    card_data = CARD_DATA_MAP.get(opp_active.id)
    if not card_data or not card_data.attacks:
        return 30
        
    max_dmg = 0
    for attack_id in card_data.attacks:
        a = ATTACK_MAP.get(attack_id)
        if a:
            dmg = a.damage
            if a.name == "Hammer-lanche":
                dmg = 300
            elif a.name == "Riptide":
                discard_pile = opponent.discard or []
                water_in_discard = sum(1 for c in discard_pile if c.id == 3)
                dmg = water_in_discard * 20
            elif a.name == "Crystal Fall":
                water_in_play = 0
                if opponent.active and opponent.active[0]:
                    water_in_play += len(opponent.active[0].energies)
                for pk in opponent.bench:
                    if pk:
                        water_in_play += len(pk.energies)
                dmg = 120 if water_in_play >= 4 else 30
            elif a.name == "Gale Thrust":
                dmg = 120
        else:
            dmg = ATTACK_DMG_MAP.get(attack_id, 0)
            
        if dmg > max_dmg:
            max_dmg = dmg
            
    return max_dmg if max_dmg > 0 else 30

def score_option(obs, opt, context, your_idx: int) -> float:
    opt_type = opt.type
    player = obs.current.players[your_idx]
    opponent = obs.current.players[1 - your_idx]
    
    score = 100.0
    card_id = get_card_id(obs, opt, your_idx)
    
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        if card_id == 722: score = 1000.0  # Snover is primary active Pokemon!
        elif card_id == 721: score = 500.0   # Kyogre is backup
        elif card_id in (31, 77, 97, 76): score = 300.0  # Fire basics
        else: score = 100.0
            
    elif context == SelectContext.SETUP_BENCH_POKEMON:
        if card_id == 722: score = 1000.0  # Snover first
        elif card_id == 721: score = 800.0   # Kyogre second
        elif card_id in (31, 77, 97, 76, 803): score = 500.0  # Others
        else: score = 100.0
            
    elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        pkmn = get_pokemon_from_option(obs, opt, your_idx)
        if pkmn:
            energy_count = len(pkmn.energies)
            if pkmn.id in (723, 46):  # Mega Abomasnow ex, Gouging Fire ex
                score = 10000.0 + energy_count * 1000.0 + pkmn.hp
            elif pkmn.id in (721, 803, 31, 583):  # Kyogre, Suicune, Chi-Yu, Keldeo ex
                score = 8000.0 + energy_count * 1000.0 + pkmn.hp
            else:
                score = 500.0 + pkmn.hp
        else:
            score = 100.0
            
    elif context == SelectContext.ATTACH_FROM:
        pkmn = get_pokemon_from_option(obs, opt, your_idx)
        if pkmn:
            energy_count = len(pkmn.energies)
            if pkmn.id in (723, 46) and energy_count < 3:
                score = 3200.0 if opt.area == AreaType.ACTIVE else 3000.0
            elif pkmn.id in (721, 803) and energy_count < 3:  # Kyogre, Suicune
                score = 3100.0 if opt.area == AreaType.ACTIVE else 2900.0
            elif pkmn.id in (31, 583) and energy_count < 2:  # Chi-Yu, Keldeo ex
                score = 2900.0 if opt.area == AreaType.ACTIVE else 2400.0
            elif pkmn.id == 722 and energy_count < 2:  # Snover
                score = 2500.0 if opt.area == AreaType.ACTIVE else 2000.0
            else:
                score = 1000.0
        else:
            score = 100.0
            
    elif context == SelectContext.ATTACH_TO:
        if card_id in (2, 3): score = 1000.0
        else: score = 100.0
            
    elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        # Check if we have Snover in play
        has_snover_in_play = False
        for pk in player.bench:
            if pk and pk.id == 722:
                has_snover_in_play = True
        if player.active and player.active[0] and player.active[0].id == 722:
            has_snover_in_play = True
            
        # Check if we have Abomasnow in hand/play
        has_abomasnow = False
        if player.active and player.active[0] and player.active[0].id == 723:
            has_abomasnow = True
        for pk in player.bench:
            if pk and pk.id == 723:
                has_abomasnow = True
        for c in (player.hand or []):
            if c.id == 723:
                has_abomasnow = True
                
        if card_id == 723:
            score = 3000.0 if has_snover_in_play else 1500.0
        elif card_id == 722:
            score = 2900.0 if (not has_snover_in_play or not has_abomasnow) else 1000.0
        elif card_id == 721:
            score = 2500.0
        elif card_id in (803, 583, 31):
            score = 2000.0
        elif card_id in (1235, 1205, 1227, 1145, 1158, 1262):
            score = 1800.0
        elif card_id in (2, 3):
            score = 1000.0
        else:
            score = 500.0
            
    elif context in (SelectContext.EVOLVES_TO, SelectContext.EVOLVES_FROM):
        if card_id == 723: score = 3000.0  # Mega Abomasnow ex
        elif card_id == 722: score = 2500.0  # Snover
        else: score = 500.0
            
    elif context in (SelectContext.ACTIVATE, SelectContext.MULLIGAN, SelectContext.COIN_HEAD, SelectContext.IS_FIRST) or opt_type in (OptionType.YES, OptionType.NO):
        if opt_type == OptionType.YES: score = 1000.0
        elif opt_type == OptionType.NO: score = 100.0
            
    elif context == SelectContext.MAIN:
        max_dmg = get_max_attack_damage(obs, your_idx)
        opp_active = opponent.active[0] if opponent.active else None
        opp_hp = opp_active.hp if opp_active else 999
        can_ko_active = (max_dmg >= opp_hp)
        
        active_pkmn = player.active[0] if player.active else None
        enemy_max_dmg = calculate_enemy_max_damage_next_turn(obs, 1 - your_idx)
        is_in_lethal_range = active_pkmn and active_pkmn.hp <= enemy_max_dmg
        
        if opt_type == OptionType.ATTACK:
            if can_ko_active: score = 15000.0 
            else: score = 7000.0 
                
        elif opt_type == OptionType.EVOLVE:
            score = 9500.0
            
        elif opt_type == OptionType.ATTACH:
            pkmn = get_pokemon_from_option(obs, opt, your_idx)
            if is_in_lethal_range and opt.inPlayArea == AreaType.ACTIVE:
                score = 3000.0
            else:
                card = None
                if opt.area == AreaType.HAND and player.hand and 0 <= opt.index < len(player.hand):
                    card = player.hand[opt.index]
                if card and pkmn:
                    if card.id in (2, 3): 
                        energy_count = len(pkmn.energies)
                        if opt.inPlayArea == AreaType.ACTIVE:
                            if pkmn.id in (723, 46) and energy_count < 3: score = 8800.0
                            elif pkmn.id in (721, 803) and energy_count < 3: score = 8780.0
                            elif pkmn.id in (31, 583) and energy_count < 2: score = 8750.0
                            elif pkmn.id == 722 and energy_count < 2: score = 8500.0
                            else: score = 1000.0
                        elif opt.inPlayArea == AreaType.BENCH:
                            if pkmn.id in (723, 46) and energy_count < 3: score = 8600.0
                            elif pkmn.id in (721, 803) and energy_count < 3: score = 8580.0
                            elif pkmn.id in (31, 583) and energy_count < 2: score = 8500.0
                            elif pkmn.id == 722 and energy_count < 2: score = 8300.0
                            else: score = 900.0
                
        elif opt_type == OptionType.PLAY:
            card = get_card_id(obs, opt, your_idx)
            
            # Check if we have Snover in play
            has_snover_in_play = False
            for pk in player.bench:
                if pk and pk.id == 722:
                    has_snover_in_play = True
            if player.active and player.active[0] and player.active[0].id == 722:
                has_snover_in_play = True
                
            if card == 1205: # Cyrano (Search ex)
                has_ex = False
                active_pkmn = player.active[0] if player.active else None
                if active_pkmn and active_pkmn.id in (46, 723):
                    has_ex = True
                for pk in player.bench:
                    if pk.id in (46, 723):
                        has_ex = True
                for c in (player.hand or []):
                    if c.id in (46, 723):
                        has_ex = True
                
                # If we don't have any ex and have Snover in play, Cyrano is top priority. Otherwise, save supporter turn for drawing.
                score = 9400.0 if (not has_ex and has_snover_in_play) else 100.0
            elif card == 1235: # Waitress (Draw)
                score = 9400.0
            elif card == 1227: # Lillie (Draw)
                score = 9300.0
            elif card == 1145: # Mega Signal
                score = 9500.0 if (is_playing_abomasnow() and has_snover_in_play) else 100.0
            elif card == 1262: # Surfing Beach
                score = 8200.0
            elif card == 1158: # Maximum Belt
                score = 8100.0
            else: score = 8000.0
                
        elif opt_type in (OptionType.ABILITY, OptionType.SKILL):
            card = get_card_id(obs, opt, your_idx)
            if card == 1262: # Surfing Beach switch skill
                score = 7800.0
            else:
                score = 1000.0
                
        elif opt_type == OptionType.RETREAT:
            score = 100.0
                
    return score

def predict_card_lists(obs: Observation):
    your_idx = obs.current.yourIndex
    player = obs.current.players[your_idx]
    opponent = obs.current.players[1 - your_idx]
    
    # 1. Calculate remaining cards in our deck & prize
    start_deck = list(_CURRENT_DECK)
    if not start_deck:
        start_deck = [3] * 60
        
    known_cards = []
    if player.active and player.active[0] is not None:
        known_cards.append(player.active[0].id)
        known_cards.extend([c.id for c in player.active[0].energyCards])
    for pk in player.bench:
        if pk is not None:
            known_cards.append(pk.id)
            known_cards.extend([c.id for c in pk.energyCards])
    if player.hand:
        known_cards.extend([c.id for c in player.hand])
    if player.discard:
        known_cards.extend([c.id for c in player.discard])
        
    remaining_pool = list(start_deck)
    for cid in known_cards:
        if cid in remaining_pool:
            remaining_pool.remove(cid)
            
    prize_count = len(player.prize)
    your_prize = remaining_pool[:prize_count]
    your_deck = remaining_pool[prize_count:]
    
    while len(your_prize) < prize_count:
        your_prize.append(3)
    while len(your_deck) < player.deckCount:
        your_deck.append(3)
    your_deck = your_deck[:player.deckCount]
    
    # 2. Opponent predictions
    opp_known = []
    if opponent.active and opponent.active[0] is not None:
        opp_known.append(opponent.active[0].id)
        opp_known.extend([c.id for c in opponent.active[0].energyCards])
    for pk in opponent.bench:
        if pk is not None:
            opp_known.append(pk.id)
            opp_known.extend([c.id for c in pk.energyCards])
    if opponent.discard:
        opp_known.extend([c.id for c in opponent.discard])
        
    opp_tracked_hand = list(_OPPONENT_HAND_TRACKED.values())
    opp_known.extend(opp_tracked_hand)
    
    opp_start_deck = detect_opponent_deck(opp_known)
    
    opp_remaining = list(opp_start_deck)
    for cid in opp_known:
        if cid in opp_remaining:
            opp_remaining.remove(cid)
            
    opp_prize_count = len(opponent.prize)
    opp_hand_count = opponent.handCount
    
    opponent_hand = list(opp_tracked_hand)[:opp_hand_count]
    needed_hand_slots = opp_hand_count - len(opponent_hand)
    opponent_hand.extend(opp_remaining[:needed_hand_slots])
    opp_remaining = opp_remaining[needed_hand_slots:]
    
    opponent_prize = opp_remaining[:opp_prize_count]
    opponent_deck = opp_remaining[opp_prize_count:]
    
    while len(opponent_prize) < opp_prize_count:
        opponent_prize.append(3)
    while len(opponent_hand) < opp_hand_count:
        opponent_hand.append(3)
    while len(opponent_deck) < opponent.deckCount:
        opponent_deck.append(3)
    opponent_deck = opponent_deck[:opponent.deckCount]
    
    opponent_active = []
    active = opponent.active
    if len(active) > 0 and active[0] is None:
        basic_id = 722 if is_playing_abomasnow() else 77
        for cid in opp_remaining:
            card_data = CARD_DATA_MAP.get(cid)
            if card_data and card_data.basic:
                basic_id = cid
                break
        opponent_active = [basic_id]
        
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.
    
    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()
        
    your_idx = obs.current.yourIndex if obs.current else 0
    update_opponent_hand_tracking(obs, your_idx)
        
    # --- PHASE 0: LOOK-AHEAD SEARCH VERIFICATION ---
    options = obs.select.option
    max_count = obs.select.maxCount
    
    if max_count == 1 and options and len(options) > 1:
        best_search_score = -9999999.0
        best_search_idx = -1
        sim_count = 0
        MAX_SIMS = 100  # Safety limit for number of simulations
        
        try:
            your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active = predict_card_lists(obs)
            
            root = search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)
            try:
                for i in range(len(options)):
                    if sim_count > MAX_SIMS:
                        break
                    try:
                        child = search_step(root.searchId, [i])
                        sim_count += 1
                        
                        child_obs = child.observation
                        child_current = child_obs.current
                        child_result = child_current.result if child_current else -1
                        
                        base_score = score_option(obs, options[i], obs.select.context, your_idx)
                        
                        # Check if this choice is a "preparation" action (e.g. we can make another choice immediately)
                        is_prep = options[i].type in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE, OptionType.ABILITY, OptionType.SKILL, OptionType.RETREAT)
                        
                        if child_result == your_idx:
                            # Immediate win
                            search_score = 9999999.0
                        elif child_result != -1:
                            # Immediate loss
                            search_score = -9999999.0
                        elif is_prep and child_obs.select and child_obs.select.option and sim_count < MAX_SIMS:
                            # Deep search (2nd step)
                            best_child2_score = -9999999.0
                            child_options = child_obs.select.option
                            
                            # Limit branching factor in step 2 if there are too many options
                            step2_options_limit = 10 if len(options) > 5 else 20
                            sorted_child_opts = []
                            for j, c_opt in enumerate(child_options):
                                try:
                                    c_score = score_option(child_obs, c_opt, child_obs.select.context, your_idx)
                                except Exception:
                                    c_score = 100.0
                                sorted_child_opts.append((c_score, j))
                            sorted_child_opts.sort(key=lambda x: x[0], reverse=True)
                            
                            for c_score, j in sorted_child_opts[:step2_options_limit]:
                                if sim_count > MAX_SIMS:
                                    break
                                try:
                                    child2 = search_step(child.searchId, [j])
                                    sim_count += 1
                                    
                                    child2_obs = child2.observation
                                    child2_current = child2_obs.current
                                    child2_result = child2_current.result if child2_current else -1
                                    
                                    if child2_result == your_idx:
                                        score2 = 9999999.0
                                    elif child2_result != -1:
                                        score2 = -9999999.0
                                    else:
                                        score2 = base_score + c_score
                                        player_before = obs.current.players[your_idx]
                                        player_after = child2_current.players[your_idx] if child2_current else None
                                        
                                        if player_after:
                                            prizes_before = sum(1 for p in player_before.prize if p is not None)
                                            prizes_after = sum(1 for p in player_after.prize if p is not None)
                                            if prizes_after < prizes_before:
                                                score2 += 5000.0 * (prizes_before - prizes_after)
                                                
                                            active_after = player_after.active if player_after else []
                                            if not active_after or active_after[0] is None:
                                                score2 -= 3000.0
                                                
                                    search_release(child2.searchId)
                                    if score2 > best_child2_score:
                                        best_child2_score = score2
                                except Exception:
                                    pass
                            
                            search_score = best_child2_score if best_child2_score > -5000000.0 else base_score
                        else:
                            # 1-step outcome (terminating action or max simulations reached)
                            search_score = base_score
                            player_before = obs.current.players[your_idx]
                            player_after = child_current.players[your_idx] if child_current else None
                            
                            if player_after:
                                prizes_before = sum(1 for p in player_before.prize if p is not None)
                                prizes_after = sum(1 for p in player_after.prize if p is not None)
                                if prizes_after < prizes_before:
                                    search_score += 5000.0 * (prizes_before - prizes_after)
                                    
                                active_after = player_after.active if player_after else []
                                if not active_after or active_after[0] is None:
                                    search_score -= 3000.0
                                    
                        search_release(child.searchId)
                        
                        if search_score > best_search_score:
                            best_search_score = search_score
                            best_search_idx = i
                    except Exception:
                        pass
            finally:
                try:
                    search_end()
                except Exception:
                    pass
            
            if best_search_idx != -1 and best_search_score > -5000000.0:
                return [best_search_idx]
        except Exception:
            pass
                
    # Attempt DRL inference if ONNX is available and loaded
    try:
        init_onnx_model()
        if _ONNX_LOADED:
            # --- PHASE 1: HYBRID HEURISTIC OVERRIDE ---
            options = obs.select.option
            max_count = obs.select.maxCount
            your_idx = obs.current.yourIndex
            
            best_score = -999999.0
            best_idx = 0
            
            if max_count == 1 and len(options) > 1:
                # Find if we have any other preparation options
                has_prep_options = any(opt.type in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE) for opt in options)
                for i, opt in enumerate(options):
                    try:
                        score = score_option(obs, opt, obs.select.context, your_idx)
                    except Exception:
                        score = 100.0
                    if score > best_score:
                        best_score = score
                        best_idx = i
                # Only override if it's a GOD MOVE and we don't have other preparation options left
                if best_score >= 12000.0 and not has_prep_options:
                    # GOD MOVE DETECTED! OVERRIDE RL!
                    return [best_idx]
            
            # --- PHASE 2: ONNX INFERENCE ---
            return run_onnx_inference(obs)
    except Exception as e:
        # Fallback to heuristics silently
        pass
    
    # --- HEURISTIC FALLBACK AGENT ---
    context = obs.select.context
    options = obs.select.option
    your_idx = obs.current.yourIndex
    
    # Score all options
    scored_options = []
    for i, opt in enumerate(options):
        try:
            score = score_option(obs, opt, context, your_idx)
        except Exception:
            score = 100.0
        scored_options.append((score, i))
        
    # Sort options by score descending
    scored_options.sort(key=lambda x: x[0], reverse=True)
    
    # Select maxCount elements
    k = obs.select.maxCount
    selected_indices = [idx for score, idx in scored_options[:k]]
    
    return selected_indices
