import os
import gymnasium as gym
import numpy as np
import random
from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, Observation, AreaType, OptionType, SelectContext
from main import score_option, read_deck_csv

MAX_OPTIONS = 50
STATE_DIM = 144

class PokemonTCGEnv(gym.Env):
    """Custom Gymnasium environment for Pokemon TCG wrapping the cabt engine."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_policy=None, player_deck=None):
        super().__init__()
        self.opponent_policy = opponent_policy if opponent_policy is not None else self._random_policy
        self.deck = player_deck if player_deck is not None else read_deck_csv()
        
        # Action space: select one of the MAX_OPTIONS options
        self.action_space = gym.spaces.Discrete(MAX_OPTIONS)
        
        # Observation space: a flat state representation vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        
        self.obs_dict = None
        self.last_prize_count = 6
        self.step_count = 0

    def _random_policy(self, obs_dict):
        """Simple random opponent fallback."""
        select_data = obs_dict.get("select")
        if not select_data:
            return []
        max_count = select_data.get("maxCount", 1)
        options_len = len(select_data.get("option", []))
        return random.sample(list(range(options_len)), max_count)

    def _extract_state(self, obs: Observation) -> np.ndarray:
        """Extract game observation into a fixed-size 1D NumPy array."""
        state = np.zeros(STATE_DIM, dtype=np.float32)
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
        for i in range(MAX_OPTIONS):
            if i < len(options):
                opt = options[i]
                state[idx] = opt.type / 16.0; idx += 1
                state[idx] = (getattr(opt, "attackId", 0) or 0) / 1200.0; idx += 1
            else:
                idx += 2 # pad with zeros
                
        return state

    def action_masks(self) -> np.ndarray:
        """Returns a boolean array where True indicates the action is legal."""
        mask = np.zeros(MAX_OPTIONS, dtype=bool)
        if not self.obs_dict or not self.obs_dict.get("select"):
            return mask
            
        options = self.obs_dict["select"]["option"]
        for i in range(min(len(options), MAX_OPTIONS)):
            mask[i] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Clean up any existing game to avoid memory leak
        try:
            battle_finish()
        except Exception:
            pass
            
        # Select opponent deck
        opponent_deck = self.deck
        if hasattr(self, 'opponent_deck_override') and self.opponent_deck_override is not None:
            opponent_deck = self.opponent_deck_override
        elif random.random() < 0.4:
            opp_decks_dir = "opponent_decks"
            if not os.path.exists(opp_decks_dir):
                opp_decks_dir = "/kaggle_simulations/agent/" + opp_decks_dir
                
            if os.path.exists(opp_decks_dir):
                csv_files = [f for f in os.listdir(opp_decks_dir) if f.endswith(".csv")]
                if csv_files:
                    chosen_csv = random.choice(csv_files)
                    chosen_path = os.path.join(opp_decks_dir, chosen_csv)
                    try:
                        with open(chosen_path, "r") as file:
                            opponent_deck = [int(line.strip()) for line in file.read().split("\n") if line.strip()][:60]
                    except Exception:
                        pass
                        
        self.obs_dict, start_data = battle_start(self.deck, opponent_deck)
        self.last_prize_count = 6
        self.step_count = 0
        
        # Fast forward if it is opponent's turn
        self._fast_forward_opponent()
        
        obs = to_observation_class(self.obs_dict)
        state_vec = self._extract_state(obs)
        
        return state_vec, {}

    def _fast_forward_opponent(self):
        """Play opponent's turns automatically until it is the Agent's turn or game finishes."""
        while True:
            if self.obs_dict is None:
                break
            current = self.obs_dict.get("current")
            if not current or current.get("result", -1) != -1:
                break
                
            active_player = current.get("yourIndex", 0)
            if active_player == 0: # Agent's turn
                break
                
            # Opponent's turn
            opponent_choices = self.opponent_policy(self.obs_dict)
            self.obs_dict = battle_select(opponent_choices)

    def step(self, action: int):
        self.step_count += 1
        
        # Get options and constraints
        if self.obs_dict is None:
            # Game ended during reset or prior transitions
            state_vec = np.zeros(STATE_DIM, dtype=np.float32)
            return state_vec, -1.0, True, False, {}
            
        select_data = self.obs_dict.get("select", {})
        options = select_data.get("option", [])
        min_count = select_data.get("minCount", 0)
        max_count = select_data.get("maxCount", 1)
        
        # Map the single discrete action chosen by RL to the list of choices
        # If action chosen is out of bounds, clip it
        primary_choice = min(int(action), len(options) - 1)
        if primary_choice < 0:
            primary_choice = 0
            
        choices = [primary_choice]
        
        # If maxCount > 1, we fill the remaining choices using our Heuristic Agent scores!
        if max_count > 1 and len(options) > 1:
            obs = to_observation_class(self.obs_dict)
            your_idx = obs.current.yourIndex
            
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
                
        # Limit to valid length
        choices = choices[:max_count]
        
        # Advance game
        self.obs_dict = battle_select(choices)
        
        # Fast-forward opponent's turn
        self._fast_forward_opponent()
        
        # Evaluate state
        if self.obs_dict is None:
            # Game ended during opponent's fast forward (we lost)
            state_vec = np.zeros(STATE_DIM, dtype=np.float32)
            return state_vec, -1.0, True, False, {}
            
        current = self.obs_dict.get("current")
        obs = to_observation_class(self.obs_dict)
        state_vec = self._extract_state(obs)
        
        # Compute Reward
        reward = 0.0
        terminated = False
        truncated = False
        
        if current:
            result = current.get("result", -1)
            if result != -1:
                terminated = True
                if result == 0:
                    reward += 1.0 # Winner reward!
                else:
                    reward -= 1.0 # Loser penalty
            else:
                # Prize card reward shaping
                player = current.get("players", [{}, {}])[0]
                prizes = player.get("prize", [])
                # Filter out None from prize cards to get remaining count
                prize_count = sum(1 for p in prizes if p is not None)
                if prize_count < self.last_prize_count:
                    # Agent took a prize card!
                    reward += 0.1 * (self.last_prize_count - prize_count)
                    self.last_prize_count = prize_count
                    
        # Small step penalty to avoid stalling
        reward -= 0.005
        
        if self.step_count >= 500:
            truncated = True
            
        return state_vec, reward, terminated, truncated, {}
