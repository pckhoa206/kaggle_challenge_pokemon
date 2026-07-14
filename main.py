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
    all_attack
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
    
    # Expand dims to batch size 1 (1, 144)
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
    if active_pkmn.id == 46: # Gouging Fire ex
        return 260 # Blaze Blitz
    elif active_pkmn.id == 31: # Chi-Yu
        return 60 # Ground Melter
    elif active_pkmn.id == 77: # Litten
        return 10
    elif active_pkmn.id == 97: # Litwick
        return 20
    elif active_pkmn.id == 76: # Slugma
        return 10
    return 10

def calculate_enemy_max_damage_next_turn(obs: Observation, opp_idx: int) -> int:
    """Safely estimate the maximum attack damage of the opponent active Pokemon next turn."""
    opponent = obs.current.players[opp_idx]
    opp_active = opponent.active[0] if opponent.active else None
    if not opp_active:
        return 0
    card_data = CARD_DATA_MAP.get(opp_active.id)
    if not card_data or not card_data.attacks:
        return 30 # default baseline damage
        
    max_dmg = 0
    for attack_id in card_data.attacks:
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
        if card_id == 46: score = 1000.0
        elif card_id == 31: score = 500.0
        else: score = 100.0
            
    elif context == SelectContext.SETUP_BENCH_POKEMON:
        if card_id == 46: score = 1000.0
        elif card_id in (31, 77, 97, 76): score = 800.0
        else: score = 100.0
            
    elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.ATTACH_FROM):
        pkmn = get_pokemon_from_option(obs, opt, your_idx)
        if pkmn:
            energy_count = len(pkmn.energies)
            if context == SelectContext.ATTACH_FROM:
                if pkmn.id == 46 and energy_count < 3:
                    score = 3200.0 if opt.area == AreaType.ACTIVE else 3000.0
                elif pkmn.id == 31 and energy_count < 2:
                    score = 2900.0 if opt.area == AreaType.ACTIVE else 2400.0
                else: score = 1000.0
            else:
                opp_active = opponent.active[0] if opponent.active else None
                opp_hp = opp_active.hp if opp_active else 999
                max_dmg = get_max_attack_damage(obs, your_idx)
                can_ko = (max_dmg >= opp_hp)
                
                if not can_ko:
                    if pkmn.id == 46 and energy_count >= 3:
                        score = 13000.0 # Strongly promote powered up main attacker!
                    elif pkmn.id == 31 and energy_count >= 2:
                        score = 11000.0
                    elif pkmn.id != 46: 
                        score = 500.0 + pkmn.hp # normal priority
                    else:
                        score = 100.0
                else:
                    if pkmn.id == 46: score = 20000.0 + energy_count * 100.0
                    elif pkmn.id == 31: score = 15000.0 + energy_count * 100.0
                    else: score = 500.0 + pkmn.hp
        else: score = 100.0
            
    elif context == SelectContext.ATTACH_TO:
        if card_id == 2: score = 1000.0
        else: score = 100.0
            
    elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        if card_id == 46: score = 3000.0
        elif card_id == 31: score = 2500.0
        elif card_id in (1235, 1205, 1227): score = 1800.0
        elif card_id == 2: score = 1000.0
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
            else: score = 10000.0 
                
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
                    if card.id == 2: 
                        energy_count = len(pkmn.energies)
                        if opt.inPlayArea == AreaType.ACTIVE:
                            if pkmn.id == 46 and energy_count < 3: score = 8800.0
                            elif pkmn.id == 31 and energy_count < 2: score = 8750.0
                            else: score = 1000.0
                        elif opt.inPlayArea == AreaType.BENCH:
                            if pkmn.id == 46 and energy_count < 3: score = 8600.0
                            elif pkmn.id == 31 and energy_count < 2: score = 8500.0
                            else: score = 900.0
                
        elif opt_type == OptionType.PLAY:
            card = get_card_id(obs, opt, your_idx)
            if card == 1205: # Cyrano (Search ex)
                has_gf_ex = False
                active_pkmn = player.active[0] if player.active else None
                if active_pkmn and active_pkmn.id == 46:
                    has_gf_ex = True
                for pkmn in player.bench:
                    if pkmn.id == 46:
                        has_gf_ex = True
                for c in (player.hand or []):
                    if c.id == 46:
                        has_gf_ex = True
                
                # If we don't have Gouging Fire ex, Cyrano is top priority. Otherwise, save supporter turn for drawing.
                score = 9400.0 if not has_gf_ex else 100.0
            elif card == 1235: # Waitress (Draw)
                score = 9400.0
            elif card == 1227: # Lillie (Draw)
                score = 9300.0
            elif card == 1145: 
                score = 100.0
            else: score = 8000.0
                
        elif opt_type == OptionType.RETREAT:
            score = 100.0
                
    return score

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
                for i, opt in enumerate(options):
                    try:
                        score = score_option(obs, opt, obs.select.context, your_idx)
                    except Exception:
                        score = 100.0
                    if score > best_score:
                        best_score = score
                        best_idx = i
                if best_score >= 8500.0:
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
